from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import Any

from trading_ml.agent_state import DecisionName, FailureCategory, LoopLimits, ReviewCheckpoint
from trading_ml.artifact_store import persist_node_artifact
from trading_ml.price_action_feature_catalog import build_strategy_intake
from trading_ml.config import load_bnr_config
from trading_ml.feature_diagnostics import build_feature_diagnostics
from trading_ml.research_program import evaluate_program_state
from trading_ml.research_controller import (
    build_feature_search_space,
    build_feature_threshold_search_space,
    build_label_search_space,
    build_model_search_space,
    build_subtype_search_space,
    build_threshold_search_space,
    build_translation_policy_search_space,
    load_controller_config,
)
from trading_ml.schemas import BlockingIssue
from trading_ml.schemas import utc_now_iso
from trading_ml.search import build_search_space, run_governed_search
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.translation_analysis import build_translation_analysis
from trading_ml.validation_audit import build_validation_audit


def _append_log(state: dict[str, Any], actor: str, message: str, payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    entry = {
        "created_at": utc_now_iso(),
        "actor": actor,
        "message": message,
        "payload": payload or {},
    }
    persist_node_artifact(
        run_id=str(state.get("run_id", "unknown-run")),
        node_name=actor,
        cycle=int(state.get("research_cycle", 1) or 1),
        phase=str(state.get("phase", "unknown")),
        state=state,
        payload=entry["payload"],
    )
    return [*state.get("run_log", []), entry]


def _ensure_counts(state: dict[str, Any]) -> dict[str, int]:
    return {
        "trials": state.get("experiment_counts", {}).get("trials", 0),
        "feature_changes": state.get("experiment_counts", {}).get("feature_changes", 0),
        "threshold_changes": state.get("experiment_counts", {}).get("threshold_changes", 0),
    }


def _append_blocking_issue(
    state: dict[str, Any],
    *,
    code: str,
    severity: str,
    category: str,
    node: str,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issues = list(state.get("blocking_issues", []))
    records = list(state.get("blocking_issue_records", []))
    if code not in {row.get("code") for row in records}:
        issues.append(message)
        records.append(
            asdict(
                BlockingIssue(
                    code=code,
                    severity=severity,
                    category=category,
                    node=node,
                    message=message,
                    evidence=evidence or {},
                )
            )
        )
    return {"blocking_issues": issues, "blocking_issue_records": records}


def _source_boundary_role(state: dict[str, Any], source_path: str | None) -> str | None:
    manifest = dict(state.get("data_manifest", {}))
    for entry in manifest.get("files", []):
        if entry.get("source_path") == source_path:
            return entry.get("boundary_role")
    return None


def strategy_intake_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    intake = build_strategy_intake(
        str(state.get("strategy_notes", "") or ""),
        dict(state.get("bnr_spec", {})),
    )
    return {
        "current_node": "strategy_intake_agent",
        "research_intake": intake,
        "run_log": _append_log(
            state,
            "strategy_intake_agent",
            "Converted strategy thesis text into a structured research backlog.",
            {
                "selected_feature_groups": intake["selected_feature_groups"],
                "next_feature_labs": intake["next_feature_labs"],
            },
        ),
    }


def program_director_node(state: dict[str, Any]) -> dict[str, Any]:
    program_state = evaluate_program_state(state)
    next_step_plan = dict(program_state.get("next_step_plan", {}))
    return {
        "current_node": "program_director",
        "program_state": program_state,
        "next_step_plan": next_step_plan,
        "run_log": _append_log(
            state,
            "program_director",
            "Reviewed research-program completeness and institutional gaps.",
            {
                "institutional_status": program_state["institutional_status"],
                "priority_mandates": program_state["priority_mandates"],
                "program_gaps": program_state["program_gaps"][:6],
                "next_step_plan": next_step_plan,
            },
        ),
    }


def governor_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    blocking_issues = list(state.get("blocking_issues", []))
    blocking_issue_records = list(state.get("blocking_issue_records", []))
    pending = list(state.get("checkpoints_pending", []))
    if state.get("phase") == "foundation" and "bnr_spec_approval" not in pending:
        pending.append("bnr_spec_approval")
    source_path = dict(state.get("stage2_config", {})).get("source_path")
    boundary_role = _source_boundary_role(state, source_path)
    if state.get("phase") != "holdout_confirmation" and boundary_role == "holdout":
        update = _append_blocking_issue(
            state,
            code="HOLDOUT_ACCESS_OUTSIDE_PHASE",
            severity="blocker",
            category="data_issue",
            node="governor_agent",
            message="Holdout source loaded outside holdout confirmation phase.",
            evidence={"source_path": source_path, "phase": state.get("phase")},
        )
        blocking_issues = update["blocking_issues"]
        blocking_issue_records = update["blocking_issue_records"]
    if state.get("holdout_consumed") and state.get("phase") not in {"holdout_confirmation", "completed"}:
        update = _append_blocking_issue(
            state,
            code="HOLDOUT_ALREADY_CONSUMED",
            severity="blocker",
            category="data_issue",
            node="governor_agent",
            message="Holdout has already been consumed; exploratory iteration is blocked.",
            evidence={"phase": state.get("phase")},
        )
        blocking_issues = update["blocking_issues"]
        blocking_issue_records = update["blocking_issue_records"]
    cycle = int(state.get("research_cycle", 1))
    max_cycles = int(state.get("max_research_cycles", 1))
    return {
        "current_node": "governor_agent",
        "checkpoints_pending": pending,
        "blocking_issues": blocking_issues,
        "blocking_issue_records": blocking_issue_records,
        "run_log": _append_log(
            state,
            "governor_agent",
            "Evaluated phase gates and evidence boundary.",
            {"research_cycle": cycle, "max_research_cycles": max_cycles},
        ),
    }


def cto_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    model_spec = dict(state.get("model_spec", {}))
    review = {
        "unified_strategy_code": True,
        "parity_test_required": True,
        "human_approval_required": "search_space_approval" in state.get("checkpoints_pending", []),
        "technical_risks": ["no live adapter yet", "no parity test fixtures yet"],
        "score_to_signal_contract_required": True,
        "shared_validation_contract_required": True,
        "active_model_family": model_spec.get("active_family", "linear_baseline"),
    }
    return {
        "current_node": "cto_agent",
        "technical_review": review,
        "run_log": _append_log(state, "cto_agent", "Reviewed architecture, parity, and safety requirements.", review),
    }


def data_steward_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    stage2_config = dict(state.get("stage2_config", {}))
    if stage2_config.get("source_path"):
        boundary_role = _source_boundary_role(state, stage2_config.get("source_path"))
        result = run_stage2_research_engine(Stage2Config(**stage2_config))
        summary = {
            "manifest_present": True,
            "timezone_expected": result["config"]["timezone"],
            "validation_status": "stage-2-run-complete",
            "data_quality": result["data_quality"],
            "zone_count": result["zone_count"],
            "candidate_count": result["candidate_count"],
            "boundary_role": boundary_role,
        }
        payload = {
            "current_node": "data_steward_agent",
            "data_summary": summary,
            "stage2_result": result,
            "holdout_consumed": bool(state.get("holdout_consumed")) or boundary_role == "holdout",
            "run_log": _append_log(state, "data_steward_agent", "Ran Stage 2 data validation and BNR research engine.", summary),
        }
        if state.get("phase") != "holdout_confirmation" and boundary_role == "holdout":
            payload.update(
                _append_blocking_issue(
                    state,
                    code="HOLDOUT_EXECUTED_OUTSIDE_PHASE",
                    severity="blocker",
                    category="data_issue",
                    node="data_steward_agent",
                    message="Holdout data execution attempted outside holdout confirmation.",
                    evidence={"source_path": stage2_config.get("source_path")},
                )
            )
        return payload
    summary = {
        "manifest_present": bool(state.get("data_manifest_loaded", False)),
        "timezone_expected": "America/New_York",
        "validation_status": "pending-stage-2-loader",
    }
    return {
        "current_node": "data_steward_agent",
        "data_summary": summary,
        "run_log": _append_log(state, "data_steward_agent", "Checked manifests and data validation prerequisites.", summary),
    }


def bnr_research_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    spec = dict(state.get("bnr_spec", {}))
    spec.setdefault("candidate_description", "Candidate setup rules pending Stage 2 engine implementation.")
    stage2 = dict(state.get("stage2_result", {}))
    if stage2:
        spec["stage2_zone_count"] = stage2.get("zone_count", 0)
        spec["stage2_candidate_count"] = stage2.get("candidate_count", 0)
        spec["sample_candidates"] = stage2.get("sample_candidates", [])
        spec["market_structure_lab"] = stage2.get("market_structure_lab", {})
    return {
        "current_node": "bnr_research_agent",
        "bnr_spec": spec,
        "candidate_setups_defined": bool(spec.get("candidate_description")),
        "run_log": _append_log(state, "bnr_research_agent", "Reviewed BNR setup definition and candidate generation scope.", spec),
    }


def labeling_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    label_spec = dict(state.get("label_spec", {}))
    label_spec.setdefault("status", "pending-human-approval")
    stage2 = dict(state.get("stage2_result", {}))
    if stage2:
        label_spec["status"] = "stage-2-labels-built"
        label_spec["label_summary"] = stage2.get("label_summary", {})
    pending = list(state.get("checkpoints_pending", []))
    approvals = dict(state.get("approvals", {}))
    if not approvals.get("label_approval", False) and "label_approval" not in pending:
        pending.append("label_approval")
    return {
        "current_node": "labeling_agent",
        "label_spec": label_spec,
        "checkpoints_pending": pending,
        "run_log": _append_log(state, "labeling_agent", "Prepared label logic and queued label review.", label_spec),
    }


def feature_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    feature_spec = dict(state.get("feature_spec", {}))
    feature_spec.setdefault("timestamp_validation", "required")
    stage2 = dict(state.get("stage2_result", {}))
    diagnostics: dict[str, Any] = {}
    if stage2:
        feature_spec["timestamp_validation"] = "complete"
        feature_spec["feature_audit"] = stage2.get("feature_audit", {})
        diagnostics = build_feature_diagnostics(stage2)
        feature_spec["feature_diagnostics_status"] = diagnostics.get("status", "pending")
    intake = dict(state.get("research_intake", {}))
    if intake.get("feature_backlog"):
        feature_spec["strategy_feature_backlog"] = intake["feature_backlog"]
    return {
        "current_node": "feature_agent",
        "feature_spec": feature_spec,
        "feature_diagnostics": diagnostics,
        "run_log": _append_log(state, "feature_agent", "Prepared feature proposal set and timestamp validation requirements.", {**feature_spec, "feature_diagnostics": diagnostics}),
    }


def model_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    model_spec = dict(state.get("model_spec", {}))
    model_spec.setdefault("status", "baseline-only")
    model_spec.setdefault("calibration_required", True)
    model_spec.setdefault("model_ladder", ["linear_baseline", "gbm", "sequence_model"])
    model_spec.setdefault("active_family", "linear_baseline")
    stage2 = dict(state.get("stage2_result", {}))
    feature_diagnostics = dict(state.get("feature_diagnostics", {}))
    controller_state = dict(state.get("controller_state", {}))
    if controller_state.get("active_model_family"):
        model_spec["active_family"] = controller_state["active_model_family"]
    if stage2:
        model_spec["status"] = stage2.get("model_summary", {}).get("status", "unknown")
        model_spec["baseline_model"] = stage2.get("model_summary", {})
        model_spec["diagnostics"] = stage2.get("model_diagnostics", {})
    if feature_diagnostics.get("status") == "complete":
        model_spec["top_features"] = feature_diagnostics.get("top_features", [])
    if model_spec.get("active_family") == "linear_baseline":
        model_spec["next_family_if_earned"] = "gbm"
    return {
        "current_node": "model_agent",
        "model_spec": model_spec,
        "run_log": _append_log(state, "model_agent", "Prepared model training and calibration plan.", model_spec),
    }


def backtest_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    summary = dict(state.get("backtest_summary", {}))
    summary.setdefault("status", "not-run")
    summary.setdefault("costs_required", True)
    summary.setdefault("slippage_required", True)
    stage2 = dict(state.get("stage2_result", {}))
    if stage2:
        label_summary = stage2.get("label_summary", {})
        summary["status"] = "label-path-simulation-complete"
        summary["candidate_count"] = stage2.get("candidate_count", 0)
        summary["avg_pnl_r"] = label_summary.get("avg_pnl_r")
        summary["positive_rate"] = label_summary.get("positive_rate")
    return {
        "current_node": "backtest_agent",
        "backtest_summary": summary,
        "run_log": _append_log(state, "backtest_agent", "Prepared simulation and metrics requirements.", summary),
    }


def search_controller_agent_node(state: dict[str, Any], limits: LoopLimits) -> dict[str, Any]:
    counts = _ensure_counts(state)
    controller = dict(state.get("controller_state", {})) or load_controller_config()
    next_step_plan = dict(state.get("next_step_plan", {}))
    if next_step_plan.get("controller_override"):
        controller.update(dict(next_step_plan["controller_override"]))
    stage2_config = dict(state.get("stage2_config", {}))
    if next_step_plan.get("stage2_overrides"):
        stage2_config.update(dict(next_step_plan["stage2_overrides"]))
    active_family = controller.get("active_family")
    search_space = dict(state.get("search_space", {})) or (
        build_model_search_space()
        if active_family == "model"
        else build_feature_search_space()
        if active_family == "feature"
        else build_feature_threshold_search_space()
        if active_family == "feature_threshold"
        else build_label_search_space()
        if active_family == "label"
        else build_subtype_search_space()
        if active_family == "subtype"
        else build_threshold_search_space()
        if active_family == "threshold"
        else build_translation_policy_search_space()
        if active_family == "translation_policy"
        else build_search_space()
    )
    approvals = dict(state.get("approvals", {}))
    pending = list(state.get("checkpoints_pending", []))
    issues = list(state.get("blocking_issues", []))
    search_results: dict[str, Any] = {}

    if not approvals.get("search_space_approval", False):
        if "search_space_approval" not in pending:
            pending.append("search_space_approval")
        return {
            "current_node": "search_controller_agent",
            "search_space": search_space,
            "checkpoints_pending": pending,
            "run_log": _append_log(
                state,
                "search_controller_agent",
                "Prepared governed search space and queued approval.",
                {"search_space": search_space},
            ),
        }

    if stage2_config.get("source_path"):
        trial_config = dict(stage2_config)
        if controller.get("active_model_family"):
            trial_config["model_family"] = controller["active_model_family"]
        search_results = run_governed_search(trial_config, controller_override=controller)
        counts["trials"] += int(search_results.get("trial_count", 0))
        family = search_results.get("family")
        if family in {"feature", "feature_threshold"}:
            counts["feature_changes"] += int(search_results.get("trial_count", 0))
        elif family in {"threshold", "translation_policy"}:
            counts["threshold_changes"] += int(search_results.get("trial_count", 0))
    blocked = counts["trials"] > limits.max_trials
    if blocked and "Max trials exceeded." not in issues:
        issues.append("Max trials exceeded.")
    return {
        "current_node": "search_controller_agent",
        "search_space": search_space,
        "search_results": search_results,
        "controller_state": controller,
        "stage2_config": stage2_config,
        "experiment_counts": counts,
        "blocking_issues": issues,
        "checkpoints_pending": [name for name in pending if name != "search_space_approval"],
        "run_log": _append_log(
            state,
            "search_controller_agent",
            "Ran governed research-controller batch." if search_results else "Evaluated constrained parameter search limits.",
            {"limits": asdict(limits), "counts": counts, "controller": controller, "next_step_plan": next_step_plan, "search_results": search_results},
        ),
    }


def audit_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    stage2 = dict(state.get("stage2_result", {}))
    if stage2:
        feature_audit = stage2.get("feature_audit", {})
        data_flags = stage2.get("data_quality", {}).get("quality_flags", [])
        robustness = _robustness_status(stage2)
        validation_audit = build_validation_audit(stage2, dict(state.get("search_results", {})), dict(state.get("controller_state", {})))
        audit_summary = {
            "leakage": "pass" if feature_audit.get("failed", 1) == 0 else "fail",
            "overfitting": validation_audit["overfitting"],
            "multiple_testing": validation_audit["multiple_testing"],
            "walk_forward": validation_audit["walk_forward"],
            "cpcv": validation_audit["cpcv"],
            "deflated_sharpe": validation_audit["deflated_sharpe"],
            "purging": validation_audit["purging"],
            "random_signal_plumbing": "pass",
            "robustness": robustness,
            "data_quality_flags": data_flags,
            "feature_diagnostics": state.get("feature_diagnostics", {}),
            "market_structure_lab": stage2.get("market_structure_lab", {}),
            "model_diagnostics": stage2.get("model_diagnostics", {}),
        }
        return {
            "current_node": "audit_agent",
            "audit_summary": audit_summary,
            "run_log": _append_log(state, "audit_agent", "Audited Stage 2 feature timestamps and data quality flags.", audit_summary),
        }
    audit_summary = {
        "leakage": "pending",
        "overfitting": "pending",
        "multiple_testing": {"status": "pending"},
        "deflated_sharpe": {"status": "pending"},
        "robustness": "pending",
    }
    return {
        "current_node": "audit_agent",
        "audit_summary": audit_summary,
        "run_log": _append_log(state, "audit_agent", "Prepared audit checklist for leakage and robustness.", audit_summary),
    }


def diagnose_failure(state: dict[str, Any]) -> FailureCategory:
    issue_records = list(state.get("blocking_issue_records", []))
    if issue_records:
        return str(issue_records[0].get("category", "unknown"))  # type: ignore[return-value]
    issues = " ".join(state.get("blocking_issues", [])).lower()
    if "manifest" in issues or "data" in issues:
        return "data_issue"
    if "feature" in issues:
        return "feature_issue"
    if "label" in issues:
        return "label_issue"
    if "model" in issues or "calibration" in issues:
        return "model_issue"
    audit_summary = dict(state.get("audit_summary", {}))
    if audit_summary.get("purging", {}).get("status") == "fail":
        return "label_issue"
    if audit_summary.get("cpcv", {}).get("status") == "fail":
        return "model_issue"
    if audit_summary.get("walk_forward", {}).get("status") == "fail":
        return "model_issue"
    if "execution" in issues or "slippage" in issues or "parity" in issues:
        return "execution_issue"
    return "unknown"


def _robustness_status(stage2: dict[str, Any]) -> str:
    data_quality = dict(stage2.get("data_quality", {}))
    flags = list(data_quality.get("quality_flags", []))
    if not flags:
        return "pending"
    non_session_flags = [flag for flag in flags if flag != "missing_regular_session_bars"]
    if non_session_flags:
        return "blocked-by-data-flags"
    if _has_opening_hours_coverage(stage2):
        return "pending"
    return "blocked-by-data-flags"


def _has_opening_hours_coverage(stage2: dict[str, Any]) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False

    config = dict(stage2.get("config", {}))
    data_quality = dict(stage2.get("data_quality", {}))
    timeframe = str(config.get("timeframe", "30s"))
    seconds_per_bar = 30 if timeframe == "30s" else 60
    latest_trigger_text = str(config.get("latest_trigger_time", "11:00:00"))
    horizon_bars = int(config.get("horizon_bars", 0))
    session_end = data_quality.get("earliest_session_end")
    if not session_end:
        return False
    session_end_ts = pd.Timestamp(session_end)
    latest_trigger = pd.Timestamp.combine(session_end_ts.date(), pd.Timestamp(latest_trigger_text).time()).tz_localize(session_end_ts.tz)
    required_end = latest_trigger + timedelta(seconds=seconds_per_bar * horizon_bars)
    return session_end_ts >= required_end


def diagnosis_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    category = diagnose_failure(state)
    diagnostics = [*state.get("diagnostics", []), {"category": category, "created_at": utc_now_iso()}]
    return {
        "current_node": "diagnosis_agent",
        "diagnostics": diagnostics,
        "run_log": _append_log(state, "diagnosis_agent", "Categorized current workflow failure.", diagnostics[-1]),
    }


def translation_checkpoint_node(state: dict[str, Any]) -> dict[str, Any]:
    bnr_config = load_bnr_config()
    stage2 = dict(state.get("stage2_result", {}))
    data_quality = dict(stage2.get("data_quality", {}))
    label_summary = dict(stage2.get("label_summary", {}))
    search_results = dict(state.get("search_results", {}))
    controller_state = dict(state.get("controller_state", {}))
    audit_summary = dict(state.get("audit_summary", {}))
    accepted_trial = dict(search_results.get("accepted_trial", {}) or {})
    candidate_count = int(stage2.get("candidate_count", 0))
    session_count = max(int(data_quality.get("sessions", 0) or 0), 1)
    breadth_per_session = candidate_count / session_count
    positive_rate = float(label_summary.get("positive_rate", 0.0) or 0.0)
    net_avg_pnl_r = accepted_trial.get("net_avg_pnl_r", search_results.get("baseline", {}).get("net_avg_pnl_r"))
    turnover_proxy = breadth_per_session
    translation_contract = dict(bnr_config.get("translation_contract", {}))
    translation_records = list(audit_summary.get("walk_forward", {}).get("stitched_prediction_records", []))
    threshold_analysis = build_translation_analysis(
        stage2,
        prediction_records=translation_records or None,
        sizing_policy=str(controller_state.get("benchmark_sizing_policy") or "binary_threshold_v1"),
        regime_throttle_policy=str(controller_state.get("benchmark_regime_throttle_policy") or "none"),
        regime_size_policy=str(controller_state.get("benchmark_regime_size_policy") or "none"),
    )
    status = threshold_analysis.get("status", "pending")
    if status == "pending" and net_avg_pnl_r is not None:
        status = "pass" if float(net_avg_pnl_r) > 0 and breadth_per_session >= float(translation_contract.get("min_breadth_per_session", 1.0)) and float(translation_contract.get("min_positive_rate", 0.05)) <= positive_rate <= float(translation_contract.get("max_positive_rate", 0.6)) else "fail"
    applied_threshold = accepted_trial.get("overrides", {}).get("decision_threshold", controller_state.get("frozen_threshold"))
    summary = {
        "status": status,
        "breadth_per_session": breadth_per_session,
        "turnover_proxy": turnover_proxy,
        "positive_rate": positive_rate,
        "net_avg_pnl_r": net_avg_pnl_r,
        "accepted_trial_id": accepted_trial.get("trial_id"),
        "threshold_analysis": threshold_analysis,
        "best_translation_row": threshold_analysis.get("best_threshold"),
        "applied_threshold": applied_threshold,
        "applied_sizing_policy": controller_state.get("benchmark_sizing_policy"),
        "applied_regime_throttle_policy": controller_state.get("benchmark_regime_throttle_policy"),
        "applied_regime_size_policy": controller_state.get("benchmark_regime_size_policy"),
    }
    if accepted_trial.get("overrides", {}).get("decision_threshold") is not None:
        summary["suggested_threshold"] = accepted_trial["overrides"]["decision_threshold"]
    elif threshold_analysis.get("status") == "pass" and controller_state.get("frozen_threshold") is None:
        best_threshold = dict(threshold_analysis.get("best_threshold", {}))
        summary["suggested_threshold"] = best_threshold.get("threshold")
    elif controller_state.get("frozen_threshold") is not None:
        summary["suggested_threshold"] = controller_state["frozen_threshold"]
    summary["suggested_sizing_policy"] = accepted_trial.get("overrides", {}).get("sizing_policy", controller_state.get("benchmark_sizing_policy"))
    summary["suggested_regime_throttle_policy"] = accepted_trial.get("overrides", {}).get("regime_throttle_policy", controller_state.get("benchmark_regime_throttle_policy"))
    summary["suggested_regime_size_policy"] = accepted_trial.get("overrides", {}).get("regime_size_policy", controller_state.get("benchmark_regime_size_policy"))
    return {
        "current_node": "translation_checkpoint",
        "translation_summary": summary,
        "run_log": _append_log(state, "translation_checkpoint", "Checked prediction-to-strategy translation quality.", summary),
    }


def promotion_decision_node(state: dict[str, Any]) -> dict[str, Any]:
    decision: DecisionName
    audit_summary = dict(state.get("audit_summary", {}))
    best_trial = dict(state.get("search_results", {}).get("best_trial", {}) or {})
    net_avg_pnl_r = best_trial.get("net_avg_pnl_r")
    translation_summary = dict(state.get("translation_summary", {}))
    model_diagnostics = dict(audit_summary.get("model_diagnostics", {}))
    calibration = dict(model_diagnostics.get("calibration_review", {}))
    walk_forward = dict(audit_summary.get("walk_forward", {}))
    cpcv = dict(audit_summary.get("cpcv", {}))
    deflated_sharpe = dict(audit_summary.get("deflated_sharpe", {}))
    purging = dict(audit_summary.get("purging", {}))
    multiple_testing = dict(audit_summary.get("multiple_testing", {}))

    if state.get("blocking_issues"):
        decision = "revise"
    elif not state.get("approvals", {}).get("frozen_spec_approval", False):
        decision = "freeze"
    elif audit_summary.get("leakage") != "pass":
        decision = "reject"
    elif purging.get("status") != "pass":
        decision = "reject"
    elif str(audit_summary.get("robustness", "")).startswith("blocked"):
        decision = "freeze"
    elif walk_forward.get("status") != "pass":
        decision = "freeze"
    elif cpcv.get("status") != "pass":
        decision = "freeze"
    elif deflated_sharpe.get("status") != "pass":
        decision = "freeze"
    elif audit_summary.get("overfitting") in {"pending", "fail"}:
        decision = "freeze"
    elif multiple_testing.get("status") != "pass":
        decision = "freeze"
    elif calibration.get("status") != "pass":
        decision = "freeze"
    elif translation_summary.get("status") == "fail":
        decision = "reject"
    elif translation_summary.get("status") != "pass":
        decision = "freeze"
    elif net_avg_pnl_r is not None and float(net_avg_pnl_r) <= 0:
        decision = "reject"
    else:
        decision = "advance_to_validation"
    return {
        "current_node": "promotion_decision",
        "promotion_decision": decision,
        "run_log": _append_log(state, "promotion_decision", "Computed promotion decision.", {"decision": decision}),
    }


def checkpoint_payload(name: str, state: dict[str, Any]) -> ReviewCheckpoint:
    instructions = {
        "bnr_spec_approval": "Review the BNR definition before candidate setup generation continues.",
        "label_approval": "Review the take / do-not-take label logic before training.",
        "search_space_approval": "Review the constrained search space before parameter iteration.",
        "frozen_spec_approval": "Approve the frozen spec before validation or promotion.",
    }
    details = {
        "bnr_spec": state.get("bnr_spec", {}),
        "label_spec": state.get("label_spec", {}),
        "search_space": state.get("search_space", {}),
        "search_results": state.get("search_results", {}),
        "translation_summary": state.get("translation_summary", {}),
        "phase": state.get("phase"),
    }
    return ReviewCheckpoint(name=name, instruction=instructions[name], details=details)


def iteration_controller_node(state: dict[str, Any]) -> dict[str, Any]:
    cycle = int(state.get("research_cycle", 1))
    max_cycles = int(state.get("max_research_cycles", 1))
    search_results = dict(state.get("search_results", {}))
    accepted_trial = dict(search_results.get("accepted_trial", {}) or {})
    controller_state = dict(state.get("controller_state", {})) or load_controller_config()
    stage2_config = dict(state.get("stage2_config", {}))
    family = str(controller_state.get("active_family", "setup"))
    translation_summary = dict(state.get("translation_summary", {}))
    continue_iteration = (
        state.get("promotion_decision") == "freeze"
        and bool(accepted_trial)
        and cycle < max_cycles
        and dict(state.get("audit_summary", {})).get("walk_forward", {}).get("status") in {"pending", "pass"}
        and dict(state.get("audit_summary", {})).get("purging", {}).get("status") != "fail"
    )
    if family == "setup" and state.get("promotion_decision") == "advance_to_validation" and bool(accepted_trial) and cycle < max_cycles:
        continue_iteration = True
    payload = {"continue_iteration": continue_iteration, "research_cycle": cycle, "max_research_cycles": max_cycles}
    if continue_iteration:
        if family == "setup" and state.get("promotion_decision") == "advance_to_validation":
            stage2_config.update(accepted_trial.get("overrides", {}))
            controller_state["active_family"] = "model"
            controller_state["active_model_family"] = "linear_baseline"
            controller_state["frozen_threshold"] = translation_summary.get("suggested_threshold", controller_state.get("frozen_threshold"))
            controller_state["spec_version"] = f"{controller_state.get('spec_version', 'bnr_spec_vA')}.model"
            payload["handoff"] = "setup_to_model"
            payload["benchmark_name"] = controller_state.get("benchmark_name")
        else:
            overrides = dict(accepted_trial.get("overrides", {}))
            if family in {"threshold", "feature_threshold", "translation_policy"}:
                controller_state["frozen_threshold"] = overrides.get("decision_threshold", controller_state.get("frozen_threshold"))
            if family == "translation_policy":
                controller_state["benchmark_sizing_policy"] = overrides.get("sizing_policy", controller_state.get("benchmark_sizing_policy"))
                controller_state["benchmark_regime_throttle_policy"] = overrides.get("regime_throttle_policy", controller_state.get("benchmark_regime_throttle_policy"))
                controller_state["benchmark_regime_size_policy"] = overrides.get("regime_size_policy", controller_state.get("benchmark_regime_size_policy"))
            stage2_overrides = {
                key: value
                for key, value in overrides.items()
                if key not in {"decision_threshold", "sizing_policy", "regime_throttle_policy", "regime_size_policy"}
            }
            if family != "threshold" and stage2_overrides:
                stage2_config.update(stage2_overrides)
            controller_state["spec_version"] = f"{controller_state.get('spec_version', 'bnr_spec_vA')}.c{cycle + 1}"
            payload["adopted_trial_id"] = accepted_trial.get("trial_id")
        payload["new_spec_version"] = controller_state["spec_version"]
        payload["stage2_config"] = stage2_config
        return {
            "current_node": "iteration_controller",
            "stage2_config": stage2_config,
            "controller_state": controller_state,
            "research_cycle": cycle + 1,
            "frozen_benchmark": {
                "benchmark_name": controller_state.get("benchmark_name"),
                "spec_version": controller_state.get("spec_version"),
                "family": controller_state.get("active_family"),
                "setup_trial_id": accepted_trial.get("trial_id"),
                "frozen_threshold": controller_state.get("frozen_threshold"),
                "sizing_policy": controller_state.get("benchmark_sizing_policy"),
                "regime_throttle_policy": controller_state.get("benchmark_regime_throttle_policy"),
                "regime_size_policy": controller_state.get("benchmark_regime_size_policy"),
            },
            "run_log": _append_log(state, "iteration_controller", "Adopted accepted exploratory trial as next frozen baseline.", payload),
        }
    return {
        "current_node": "iteration_controller",
        "run_log": _append_log(state, "iteration_controller", "Stopped autonomous research cycling.", payload),
    }
