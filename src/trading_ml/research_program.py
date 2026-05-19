from __future__ import annotations

from typing import Any

from trading_ml.config import load_research_program_config


def build_program_state() -> dict[str, Any]:
    config = load_research_program_config()["program"]
    return {
        "name": config["name"],
        "mandate": config["mandate"],
        "primary_objective": config["primary_objective"],
        "promotion_standard": config["promotion_standard"],
        "principles": list(config["principles"]["items"]),
        "market_structure_questions": list(config["market_structure_questions"]["items"]),
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
    program["workstreams"] = workstreams
    program["program_gaps"] = program_gaps
    program["priority_mandates"] = priority_mandates
    program["next_step_plan"] = next_step_plan
    program["institutional_status"] = "ready_for_confirmation" if not program_gaps else "research_os_incomplete"
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
    if any("regime" in name for name in feature_diagnostics.get("family_scores", {}).keys()):
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
    if audit.get("random_signal_plumbing") not in (None, "pending"):
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
        mandates.append("Build a BNR failure taxonomy from false positives, false negatives, and no-trade cases.")
    if "model_lab: missing bucket_monotonicity" in program_gaps:
        mandates.append("Add prediction bucket monotonicity and score-decile diagnostics before more model escalation.")
    if "validation_lab: missing random_signal_plumbing" in program_gaps:
        mandates.append("Keep a plumbing-control backtest in the standard workflow for every promoted benchmark.")
    if "portfolio_lab: missing position_sizing_policy" in program_gaps:
        mandates.append("Formalize score-to-size translation before calling the system institutional-grade.")
    if "production_lab: missing shadow_mode_plan" in program_gaps:
        mandates.append("Define shadow-mode and rollback requirements now, before widening autonomous search.")
    if audit.get("walk_forward") == "fail":
        mandates.append("Treat market-structure and label quality as the next research lever, not threshold churn.")
    if min_sessions and sessions and sessions < min_sessions:
        mandates.append(f"Do not trust exploratory winners below the {min_sessions}-session sample floor.")
    if translation.get("status") == "pass":
        mandates.append("Preserve execution realism and utility scoring as hard gates in every cycle.")
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
    top_features = [row.get("feature") for row in shap.get("top_features", []) if row.get("feature")]
    utility_status = translation.get("status")
    best_translation = dict(translation.get("best_threshold", {}))
    utility_gap = best_translation.get("utility_gap_vs_binary")

    if walk_forward.get("status") == "pass" and cpcv.get("status") == "fail":
        filter_family = "policy_meta"
        if any(name in top_features for name in ["reclaim_body_strength", "post_reclaim_close_strength", "reclaim_close_location"]):
            filter_family = "policy_meta"
        elif any(name in top_features for name in ["break_close_distance_to_zone", "break_efficiency_ratio", "first_break_close_excess_points"]):
            filter_family = "policy_gate"
        return {
            "status": "ready",
            "lane": "robustness_rebuild",
            "action": "run_targeted_policy_filter_cycle",
            "reason": "Walk-forward passes but CPCV fails; improve robustness before model escalation.",
            "controller_override": {"active_family": filter_family},
            "stage2_overrides": {"feature_family": benchmark.get("feature_family") or state.get("stage2_config", {}).get("feature_family")},
            "policy_family": filter_family,
            "benchmark_contract": benchmark,
            "success_criteria": ["cpcv status = pass", "mean_total_pnl_r > 0", "walk_forward stays pass"],
        }
    if walk_forward.get("status") == "pass" and cpcv.get("status") == "pass" and deflated_sharpe.get("status") == "fail":
        return {
            "status": "ready",
            "lane": "research_cycle",
            "action": "reduce_search_complexity_and_retest",
            "reason": "Validation is acceptable, but DSR says observed performance is not yet above multiple-testing inflation.",
            "controller_override": {"active_family": "translation_policy"},
            "stage2_overrides": {"feature_family": benchmark.get("feature_family") or state.get("stage2_config", {}).get("feature_family")},
            "benchmark_contract": benchmark,
            "success_criteria": ["deflated sharpe probability >= 0.95", "walk_forward stays pass", "cpcv stays pass"],
        }
    if utility_status == "pass" and calibration.get("status") == "fail" and benchmark.get("sizing_policy") not in {None, "binary_threshold_v1"}:
        return {
            "status": "ready",
            "lane": "translation_lab",
            "action": "fallback_to_conservative_sizing",
            "reason": "Utility passes, but calibration is weak for probability-driven sizing.",
            "controller_override": {"active_family": "translation_policy"},
            "stage2_overrides": {"feature_family": benchmark.get("feature_family") or state.get("stage2_config", {}).get("feature_family")},
            "benchmark_contract": benchmark,
            "success_criteria": ["binary or tiered sizing matches or beats current utility", "walk_forward stays pass", "cpcv stays pass"],
        }
    if walk_forward.get("status") == "fail":
        return {
            "status": "ready",
            "lane": "feature_lab",
            "action": "run_market_structure_feature_cycle",
            "reason": "Generalization is weak; improve the setup representation before more policy or model churn.",
            "controller_override": {"active_family": "feature"},
            "stage2_overrides": {"feature_family": benchmark.get("feature_family") or state.get("stage2_config", {}).get("feature_family")},
            "benchmark_contract": benchmark,
            "success_criteria": ["walk_forward status = pass", "utility improves versus frozen benchmark"],
        }
    if walk_forward.get("status") == "pass" and cpcv.get("status") == "pass" and utility_status == "pass":
        return {
            "status": "ready",
            "lane": "validation_lab",
            "action": "advance_to_reserved_validation_window",
            "reason": "Exploration benchmark is robust enough to test on untouched validation data.",
            "controller_override": {"active_family": "validation_window"},
            "stage2_overrides": {},
            "benchmark_contract": benchmark,
            "success_criteria": ["validation utility positive", "validation CPCV pass or not applicable", "no benchmark edits during validation"],
        }
    if walk_forward.get("status") == "pass" and cpcv.get("status") == "pass" and utility_status == "fail":
        return {
            "status": "ready",
            "lane": "translation_lab",
            "action": "run_translation_policy_cycle",
            "reason": "Robustness is acceptable, but score-to-trade translation is weak.",
            "controller_override": {"active_family": "translation_policy"},
            "stage2_overrides": {"feature_family": benchmark.get("feature_family") or state.get("stage2_config", {}).get("feature_family")},
            "benchmark_contract": benchmark,
            "success_criteria": ["utility improves versus frozen benchmark", "walk_forward stays pass", "cpcv stays pass"],
        }
    if utility_gap is not None and float(utility_gap) > 3.0:
        return {
            "status": "ready",
            "lane": "translation_lab",
            "action": "audit_sizing_lift",
            "reason": "Sized utility materially exceeds binary utility; confirm the gain is not mostly sizing-driven.",
            "controller_override": {"active_family": "translation_policy"},
            "stage2_overrides": {"feature_family": benchmark.get("feature_family") or state.get("stage2_config", {}).get("feature_family")},
            "benchmark_contract": benchmark,
            "success_criteria": ["sizing lift is stable", "binary utility remains positive", "calibration is acceptable for probability-based sizing"],
        }
    return {
        "status": "ready",
        "lane": "research_cycle",
        "action": "continue_governed_benchmark_iteration",
        "reason": "No higher-priority protocol failure detected.",
        "controller_override": controller,
        "stage2_overrides": {},
        "benchmark_contract": benchmark,
        "success_criteria": ["improvement beats frozen benchmark on utility and robustness"],
    }
