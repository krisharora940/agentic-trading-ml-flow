from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from trading_ml.market_state_quality import build_market_state_setup_quality_diagnostic
from trading_ml.paths import REPORTS_DIR
from trading_ml.research_controller import run_governed_research_cycle
from trading_ml.setup_redesign import build_setup_redesign_plan


ResearchActionFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ResearchActionSpec:
    action_id: str
    family: str
    description: str
    callable_kind: str
    bounded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def available_research_actions() -> dict[str, ResearchActionSpec]:
    return {
        "domain_prior_ingestion": ResearchActionSpec("domain_prior_ingestion", "research_foundation", "Ingest curated BNR domain priors into measurable hypotheses.", "stateful_diagnostic_action"),
        "setup": ResearchActionSpec("setup", "setup", "Governed setup-geometry search batch.", "governed_research_cycle"),
        "model": ResearchActionSpec("model", "model", "Governed model-family search batch.", "governed_research_cycle"),
        "feature": ResearchActionSpec("feature", "feature", "Governed feature-family search batch.", "governed_research_cycle"),
        "feature_threshold": ResearchActionSpec("feature_threshold", "feature_threshold", "Governed feature plus threshold batch.", "governed_research_cycle"),
        "threshold": ResearchActionSpec("threshold", "threshold", "Governed threshold-only search batch.", "governed_research_cycle"),
        "translation_policy": ResearchActionSpec("translation_policy", "translation_policy", "Governed score-to-trade translation batch.", "governed_research_cycle"),
        "label": ResearchActionSpec("label", "label", "Governed label-policy search batch.", "governed_research_cycle"),
        "sample_expansion": ResearchActionSpec("sample_expansion", "sample_expansion", "Governed sample-expansion batch.", "governed_research_cycle"),
        "subtype": ResearchActionSpec("subtype", "subtype", "Callable subtype cycle selected by the research director.", "governed_research_cycle"),
        "policy_gate": ResearchActionSpec("policy_gate", "policy_gate", "Callable break-quality gate cycle selected by the research director.", "governed_research_cycle"),
        "policy_meta": ResearchActionSpec("policy_meta", "policy_meta", "Callable reclaim/meta filter cycle selected by the research director.", "governed_research_cycle"),
        "tail_path_cleanup": ResearchActionSpec("tail_path_cleanup", "tail_path_cleanup", "Callable CPCV tail-path cleanup cycle.", "governed_research_cycle"),
        "market_state_setup_quality": ResearchActionSpec("market_state_setup_quality", "market_state_setup_quality", "Callable market-state/setup-quality cycle.", "governed_research_cycle"),
        "exit_behavior_research": ResearchActionSpec("exit_behavior_research", "exit_behavior_research", "Callable exit-behavior research cycle.", "governed_research_cycle"),
        "candidate_universe_expansion": ResearchActionSpec("candidate_universe_expansion", "candidate_universe_expansion", "Callable candidate-universe expansion cycle.", "governed_research_cycle"),
        "setup_redesign": ResearchActionSpec("setup_redesign", "setup", "Prepare a BNR setup-redesign mandate from repeated structural failure.", "stateful_diagnostic_action"),
        "validation_failure_analysis": ResearchActionSpec("validation_failure_analysis", "research_diagnostics", "Summarize why the current BNR candidate failed validation and what to try next.", "stateful_diagnostic_action"),
        "cpcv_attribution": ResearchActionSpec("cpcv_attribution", "research_diagnostics", "Load CPCV failure attribution as a callable director diagnostic.", "stateful_diagnostic_action"),
        "ml4t_backtest": ResearchActionSpec("ml4t_backtest", "research_diagnostics", "Replay the current governed benchmark through the ML4T event-driven backtest path.", "stateful_diagnostic_action"),
        "validation_window": ResearchActionSpec("validation_window", "validation_window", "Callable reserved-validation confirmation cycle.", "governed_research_cycle"),
        "holdout_confirmation": ResearchActionSpec("holdout_confirmation", "holdout_confirmation", "Callable holdout confirmation cycle.", "governed_research_cycle"),
    }


def execute_research_action(
    action_id: str,
    *,
    base_config: dict[str, Any],
    controller_state: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = available_research_actions()
    spec = registry.get(action_id)
    if spec is None:
        raise ValueError(f"Unknown research action: {action_id}")
    if spec.callable_kind == "governed_research_cycle":
        result = run_governed_research_cycle(
            base_config,
            family=spec.family,
            controller_override=controller_state,
        )
    else:
        result = _execute_stateful_action(
            action_id,
            state=state or {},
            base_config=base_config,
            controller_state=controller_state,
        )
    return {
        **result,
        "action": spec.to_dict(),
    }


def _execute_stateful_action(
    action_id: str,
    *,
    state: dict[str, Any],
    base_config: dict[str, Any],
    controller_state: dict[str, Any],
) -> dict[str, Any]:
    if action_id == "domain_prior_ingestion":
        from trading_ml.research_os import build_curated_domain_priors, build_hypotheses_from_priors, build_research_backlog

        priors = build_curated_domain_priors()
        hypotheses = build_hypotheses_from_priors(priors)
        backlog = build_research_backlog(
            hypotheses,
            list(state.get("failure_memory", []) or []),
            stage2_result=dict(state.get("stage2_result", {}) or {}),
        )
        return {
            "family": "research_foundation",
            "status": "complete",
            "trial_count": 0,
            "batch_decision": "inform",
            "domain_priors": priors,
            "research_backlog": backlog,
            "active_hypothesis": dict(backlog[0]) if backlog else {},
        }
    if action_id == "setup_redesign":
        plan = build_setup_redesign_plan(state)
        diagnostic = build_market_state_setup_quality_diagnostic(state)
        return {
            "family": "setup",
            "status": "complete",
            "trial_count": 0,
            "batch_decision": "inform",
            "setup_redesign_plan": {**plan, "market_state_setup_quality_diagnostic": diagnostic},
            "market_state_setup_quality_diagnostic": diagnostic,
        }
    if action_id == "validation_failure_analysis":
        return _validation_failure_analysis(state)
    if action_id == "cpcv_attribution":
        return _cpcv_attribution()
    if action_id == "ml4t_backtest":
        from trading_ml.ml4t_backtest_adapter import run_market_state_v1_ml4t_backtest

        boundary_role = str(controller_state.get("boundary_role", "exploration") or "exploration")
        bundle = run_market_state_v1_ml4t_backtest(boundary_role=boundary_role)
        return {
            "family": "research_diagnostics",
            "status": "complete",
            "trial_count": 1,
            "batch_decision": "inform",
            "ml4t_backtest": bundle.report,
            "artifacts": {
                "output_path": str(bundle.output_path),
                "run_dir": str(bundle.run_dir),
            },
        }
    raise ValueError(f"Unsupported stateful research action: {action_id}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _cpcv_attribution() -> dict[str, Any]:
    attribution = _read_json(REPORTS_DIR / "cpcv_failure_attribution.json")
    return {
        "family": "research_diagnostics",
        "status": "complete" if attribution else "missing",
        "trial_count": 0,
        "batch_decision": "inform",
        "cpcv_attribution": attribution,
    }


def _validation_failure_analysis(state: dict[str, Any]) -> dict[str, Any]:
    audit = dict(state.get("audit_summary", {}) or {})
    translation = dict(state.get("translation_summary", {}) or {})
    active_hypothesis = dict(state.get("active_hypothesis", {}) or {})
    cpcv = _read_json(REPORTS_DIR / "cpcv_failure_attribution.json")
    latest_failure = dict((state.get("failure_memory", []) or [])[-1] if state.get("failure_memory") else {})
    report = {
        "status": "complete",
        "hypothesis_id": active_hypothesis.get("hypothesis_id"),
        "family": active_hypothesis.get("family"),
        "failure_type": latest_failure.get("failure_type"),
        "promotion_decision": state.get("promotion_decision"),
        "current_blockers": list(state.get("blocking_issues", []) or []),
        "walk_forward_status": dict(audit.get("walk_forward", {}) or {}).get("status"),
        "cpcv_status": dict(audit.get("cpcv", {}) or {}).get("status"),
        "deflated_sharpe_status": dict(audit.get("deflated_sharpe", {}) or {}).get("status"),
        "multiple_testing_status": dict(audit.get("multiple_testing", {}) or {}).get("status"),
        "calibration_status": dict(audit.get("model_diagnostics", {}) or {}).get("calibration_review", {}).get("status"),
        "translation_status": translation.get("status"),
        "dominant_failure_axes": dict(cpcv.get("dominant_failure_axes", {}) or {}),
        "recommended_next_actions": _recommended_next_actions(state, cpcv),
    }
    return {
        "family": "research_diagnostics",
        "status": "complete",
        "trial_count": 0,
        "batch_decision": "inform",
        "validation_failure_analysis": report,
    }


def _recommended_next_actions(state: dict[str, Any], cpcv: dict[str, Any]) -> list[str]:
    recent_failure = dict((state.get("failure_memory", []) or [])[-1] if state.get("failure_memory") else {})
    failure_type = str(recent_failure.get("failure_type", "") or "")
    actions: list[str] = []
    if failure_type == "cpcv_tail_path_fragility":
        actions.extend(["candidate_universe_expansion", "exit_behavior_research"])
    if dict(cpcv.get("dominant_failure_axes", {}) or {}).get("subtype"):
        actions.append("setup_redesign")
    if not actions:
        actions.append("domain_prior_ingestion")
    return actions
