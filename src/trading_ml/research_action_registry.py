from __future__ import annotations

from typing import Any

from trading_ml.research_actions import (
    available_research_actions,
    execute_research_action,
)
from trading_ml.schemas import ResearchActionPlan, ResearchActionResult, utc_now_iso


PROPOSAL_FAMILY_ACTIONS = {
    "feature": "market_state_setup_quality",
    "setup": "market_state_setup_quality",
    "eligibility": "candidate_universe_expansion",
    "path_modeling": "exit_behavior_research",
    "exit_behavior_research": "exit_behavior_research",
}


def available_action_registry() -> dict[str, dict[str, Any]]:
    return {
        action_id: spec.to_dict()
        for action_id, spec in available_research_actions().items()
    }


def action_id_for_proposal_family(family: str) -> str:
    return PROPOSAL_FAMILY_ACTIONS.get(family, "market_state_setup_quality")


def build_research_action_plan(
    proposal: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    family = str(proposal.get("family", "") or "feature")
    action_id = str(
        proposal.get("action_id", "") or action_id_for_proposal_family(family)
    )
    cluster_id = proposal.get("target_failure_cluster") or proposal.get(
        "derived_from_cluster"
    )
    controller_state = {
        "active_family": available_research_actions()
        .get(action_id, available_research_actions()["feature"])
        .family,
        "boundary_role": "exploration",
        "proposal_id": proposal.get("proposal_id"),
        "proposal_family": family,
        "target_failure_cluster": cluster_id,
        "target_market_state": proposal.get("target_market_state"),
        "target_setup_state": proposal.get("target_setup_state"),
        "target_environment_state": proposal.get("target_environment_state"),
        "target_path_class": proposal.get("target_path_class"),
        "validation_authority": "governor_graph",
    }
    plan = ResearchActionPlan(
        plan_id=f"RAP-{utc_now_iso()}",
        proposal_id=str(proposal.get("proposal_id", "")),
        action_id=action_id,
        family=family,
        objective=str(
            proposal.get("claim")
            or proposal.get("hypothesis")
            or "market-state research action"
        ),
        target_failure_cluster=cluster_id,
        expected_metric_delta=dict(
            proposal.get("expected_metric_delta", {}) or _default_metric_delta(family)
        ),
        allowable_knobs=list(
            proposal.get("allowable_knobs", []) or _default_allowable_knobs(family)
        ),
        forbidden_knobs=list(
            proposal.get("forbidden_knobs", []) or _default_forbidden_knobs()
        ),
        support_requirements=list(
            proposal.get("support_requirements", [])
            or _default_support_requirements(family)
        ),
        falsification_rule=str(
            proposal.get("falsification_rule", "")
            or "Reject if the expected robustness or continuation-validity delta does not appear in exploration evidence."
        ),
        kill_criteria=list(
            proposal.get("kill_criteria", []) or _default_kill_criteria(family)
        ),
        controller_state={
            key: value for key, value in controller_state.items() if value is not None
        },
        base_config_overrides=dict(state.get("stage2_config_overrides", {}) or {}),
    )
    return plan.to_dict()


def execute_research_action_plan(
    plan: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    action_id = str(plan.get("action_id", "") or "")
    proposal_id = str(plan.get("proposal_id", "") or "")
    if not action_id:
        return _result(
            action_id="", proposal_id=proposal_id, status="skipped_missing_action"
        )
    base_config = _base_config_from_state(state)
    if not base_config:
        return _result(
            action_id=action_id,
            proposal_id=proposal_id,
            status="skipped_missing_base_config",
            raw_summary={
                "reason": "No stage2_config or stage2_result.config was available."
            },
        )
    raw = execute_research_action(
        action_id,
        base_config={
            **base_config,
            **dict(plan.get("base_config_overrides", {}) or {}),
        },
        controller_state=dict(plan.get("controller_state", {}) or {}),
        state=state,
    )
    return _result(
        action_id=action_id,
        proposal_id=proposal_id,
        status=str(raw.get("status", "complete") or "complete"),
        family=str(raw.get("family", "") or ""),
        trial_count=int(raw.get("trial_count", 0) or 0),
        batch_decision=str(raw.get("batch_decision", "") or ""),
        metrics=_extract_metrics(raw),
        artifacts=dict(raw.get("artifacts", {}) or {}),
        raw_summary=raw,
    )


def _base_config_from_state(state: dict[str, Any]) -> dict[str, Any]:
    stage2_config = dict(state.get("stage2_config", {}) or {})
    if stage2_config:
        return stage2_config
    return dict(dict(state.get("stage2_result", {}) or {}).get("config", {}) or {})


def _extract_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    best = dict(raw.get("best_trial", {}) or {})
    accepted = dict(raw.get("accepted_trial", {}) or {})
    source = accepted or best
    return {
        key: source.get(key)
        for key in (
            "net_delta_vs_baseline",
            "roc_auc_delta_vs_baseline",
            "net_avg_pnl_r",
            "roc_auc",
            "cpcv_delta",
            "dsr_delta",
            "calibration_delta",
            "worst_path_loss_delta",
        )
        if key in source
    }


def _default_metric_delta(family: str) -> dict[str, Any]:
    return {
        "primary": "auction_state_continuation_validity",
        "cpcv_pbo": "decrease",
        "worst_path_loss": "improve",
        "calibration": "non_degrading",
        "trade_count": "support_floor_respected",
        "family": family,
    }


def _default_allowable_knobs(family: str) -> list[str]:
    mapping = {
        "feature": ["state_ontology_feature_group", "continuation_profile_feature"],
        "setup": ["state_conditioned_eligibility_axis"],
        "eligibility": ["state_taxonomy_axis", "failure_cluster_filter"],
        "path_modeling": ["continuation_profile_axis", "failure_profile_axis"],
        "exit_behavior_research": ["failure_state_exit_family"],
    }
    return mapping.get(family, ["state_conditioned_research_axis"])


def _default_forbidden_knobs() -> list[str]:
    return [
        "holdout_access",
        "future_information",
        "path_specific_tuning",
        "tiny_time_bucket_tuning",
        "unbounded_threshold_search",
        "self_modifying_strategy_code",
    ]


def _default_support_requirements(family: str) -> list[str]:
    return [
        "decision_time_available_inputs",
        "minimum_state_sample_support",
        "artifact_written_by_deterministic_executor",
        f"family_scope:{family}",
    ]


def _default_kill_criteria(family: str) -> list[str]:
    return [
        f"kill {family} branch after two accepted cycles without robustness improvement",
        "kill if state-conditioned support falls below sample floor",
        "kill if improvement is local-only and fails CPCV or DSR review",
    ]


def _result(
    *,
    action_id: str,
    proposal_id: str,
    status: str,
    family: str = "",
    trial_count: int = 0,
    batch_decision: str = "",
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    raw_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return ResearchActionResult(
        result_id=f"RAR-{utc_now_iso()}",
        action_id=action_id,
        proposal_id=proposal_id,
        status=status,
        family=family,
        trial_count=trial_count,
        batch_decision=batch_decision,
        metrics=metrics or {},
        artifacts=artifacts or {},
        raw_summary=raw_summary or {},
    ).to_dict()
