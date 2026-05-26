from __future__ import annotations

from typing import Any

from trading_ml.schemas import (
    FamilyExhaustionRecord,
    MarginalImprovementTracker,
    ResearchBranchStatus,
)


RESPONSIBILITY_BOUNDARIES: dict[str, dict[str, list[str]]] = {
    "research_desk": {
        "allowed": [
            "hypothesis_generation",
            "ontology_building",
            "feature_proposals",
            "state_taxonomy",
            "failure_clustering",
            "action_proposals",
        ],
        "forbidden": [
            "deterministic_execution",
            "metric_evaluation",
            "validation_authority",
            "promotion_authority",
            "holdout_authority",
            "search_budget_authority",
        ],
    },
    "research_executor": {
        "allowed": [
            "deterministic_execution",
            "bounded_experiments",
            "artifact_generation",
            "metric_computation",
        ],
        "forbidden": [
            "hypothesis_generation",
            "promotion_authority",
            "holdout_authority",
            "validation_authority",
        ],
    },
    "governor": {
        "allowed": [
            "validation_authority",
            "promotion_authority",
            "holdout_authority",
            "search_budget_authority",
        ],
        "forbidden": [
            "feature_proposals",
            "freeform_experimentation",
            "self_modifying_strategy_code",
        ],
    },
    "llm": {
        "allowed": ["synthesize", "propose", "prioritize", "interpret"],
        "forbidden": [
            "validate",
            "compute_metrics",
            "promote",
            "control_holdout",
            "execute_experiments",
        ],
    },
}

PROPOSAL_KEYS = {"desk_proposals", "research_action_plan"}
AUTHORITY_KEYS = {
    "promotion_decision",
    "holdout_consumed",
    "approvals",
    "validation_decision",
    "search_budget_decision",
}


def build_responsibility_boundary_summary() -> dict[str, Any]:
    return {
        "status": "active",
        "rule": "No single node may both propose and approve.",
        "boundaries": RESPONSIBILITY_BOUNDARIES,
        "state_first_modeling_doctrine": {
            "primary_modeling_target": "auction_state_continuation_validity",
            "bnr_role": "event_trigger_within_state_machine",
            "forbidden_target": "BNR_as_standalone_setup_optimizer",
        },
    }


def validate_node_responsibility(
    node_name: str, produced_keys: set[str]
) -> dict[str, Any]:
    proposes = bool(PROPOSAL_KEYS & produced_keys)
    approves = bool(AUTHORITY_KEYS & produced_keys)
    violations: list[str] = []
    if proposes and approves:
        violations.append("node_mixes_proposal_and_authority")
    if (
        node_name in {"price_action_expert", "feature_engineer", "setup_spec_agent"}
        and approves
    ):
        violations.append("research_desk_node_claimed_authority")
    if node_name == "action_executor" and proposes:
        violations.append("executor_node_generated_research_proposal")
    return {
        "node": node_name,
        "status": "pass" if not violations else "fail",
        "violations": violations,
        "produced_keys": sorted(produced_keys),
    }


def build_research_branch_statuses(
    state: dict[str, Any],
    *,
    families: list[str] | None = None,
    max_same_family_cycles: int = 2,
    max_accepted_without_robustness: int = 2,
) -> list[dict[str, Any]]:
    family_names = families or [
        "feature",
        "setup",
        "eligibility",
        "path_modeling",
        "exit_behavior_research",
    ]
    action_history = list(state.get("research_action_history", []) or [])
    desk_memory = list(state.get("desk_memory", []) or [])
    latest_evidence = dict(state.get("marginal_evidence", {}) or {})
    return [
        _branch_status(
            family,
            action_history=action_history,
            desk_memory=desk_memory,
            latest_evidence=latest_evidence,
            max_same_family_cycles=max_same_family_cycles,
            max_accepted_without_robustness=max_accepted_without_robustness,
        )
        for family in family_names
    ]


def _branch_status(
    family: str,
    *,
    action_history: list[dict[str, Any]],
    desk_memory: list[dict[str, Any]],
    latest_evidence: dict[str, Any],
    max_same_family_cycles: int,
    max_accepted_without_robustness: int,
) -> dict[str, Any]:
    governed_family = _governed_family(family)
    consecutive_cycles = 0
    for row in reversed(action_history):
        if str(row.get("family")) == governed_family:
            consecutive_cycles += 1
            continue
        break
    accepted = [
        row
        for row in action_history
        if str(row.get("family")) == governed_family
        and str(row.get("batch_decision")) == "accept"
    ]
    recent_accepted = accepted[-max_accepted_without_robustness:]
    accepted_without_robustness = sum(
        1 for row in recent_accepted if not _has_robustness_improvement(row)
    )
    last_proposal = next(
        (
            row
            for row in reversed(desk_memory)
            if str(row.get("proposal_family")) == family
        ),
        {},
    )
    latest_tracker = MarginalImprovementTracker(
        family=family,
        proposal_id=str(
            latest_evidence.get("proposal_id") or last_proposal.get("proposal_id") or ""
        ),
        action_id=str(latest_evidence.get("action_id") or governed_family),
        metric_deltas={
            key: latest_evidence.get(key)
            for key in (
                "net_delta_vs_baseline",
                "robustness_delta",
                "cpcv_delta",
                "dsr_delta",
                "calibration_delta",
                "worst_path_loss_delta",
            )
            if latest_evidence.get(key) is not None
        },
        has_robustness_improvement=_evidence_has_robustness(latest_evidence),
        decision=str(latest_evidence.get("decision", "inform") or "inform"),
    )
    exhausted = (
        consecutive_cycles >= max_same_family_cycles
        or accepted_without_robustness >= max_accepted_without_robustness
    )
    reason = ""
    if consecutive_cycles >= max_same_family_cycles:
        reason = "same_family_cycle_limit_reached"
    elif accepted_without_robustness >= max_accepted_without_robustness:
        reason = "accepted_without_robustness_limit_reached"
    exhaustion = FamilyExhaustionRecord(
        family=family,
        status="exhausted" if exhausted else "open",
        reason=reason,
        consecutive_cycles=consecutive_cycles,
        accepted_without_robustness=accepted_without_robustness,
        last_proposal_id=last_proposal.get("proposal_id"),
    )
    return ResearchBranchStatus(
        family=family,
        status="exhausted" if exhausted else "open",
        exhaustion=exhaustion.to_dict(),
        marginal_improvement=latest_tracker.to_dict(),
        next_allowed_action="" if exhausted else governed_family,
        notes=(
            ["Route exhausted families away from further proposal generation."]
            if exhausted
            else []
        ),
    ).to_dict()


def _governed_family(proposal_family: str) -> str:
    return {
        "eligibility": "candidate_universe_expansion",
        "path_modeling": "exit_behavior_research",
        "exit_behavior_research": "exit_behavior_research",
        "setup": "market_state_setup_quality",
        "feature": "market_state_setup_quality",
    }.get(proposal_family, proposal_family)


def _has_robustness_improvement(row: dict[str, Any]) -> bool:
    evidence = dict(row.get("marginal_evidence", {}) or {})
    return _evidence_has_robustness(evidence)


def _evidence_has_robustness(evidence: dict[str, Any]) -> bool:
    return bool(
        evidence.get("robustness_delta")
        or evidence.get("cpcv_delta")
        or evidence.get("dsr_delta")
        or evidence.get("calibration_delta")
        or evidence.get("worst_path_loss_delta")
    )
