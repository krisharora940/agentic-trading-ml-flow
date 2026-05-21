from __future__ import annotations

from typing import Any

from trading_ml.config import load_research_program_config
from trading_ml.research_actions import available_research_actions
from trading_ml.schemas import utc_now_iso


def build_curated_domain_priors() -> list[dict[str, Any]]:
    config = load_research_program_config()["program"]
    sources = list(config.get("domain_research", {}).get("priority_sources", []))
    default_source = sources[0] if sources else "ml4trading.io"
    priors = [
        {
            "prior_id": "DP-0001",
            "source": default_source,
            "topic": "opening_auction_state",
            "claim": "Opening auction state changes whether a breakout should be treated as continuation or repair.",
            "measurable_translation": "Classify the first 30-60 minutes into balance, trend, squeeze-release, and repair states available at decision time.",
            "family": "setup",
            "testability": "high",
            "pit_risk": "medium",
            "tags": ["bnr", "opening-auction", "market-structure"],
        },
        {
            "prior_id": "DP-0002",
            "source": default_source,
            "topic": "candidate_universe",
            "claim": "The correct unit of opportunity is often a structure attempt, not a session-level binary label.",
            "measurable_translation": "Compare session-first, attempt-first, and structure-first candidate lineages under deduplication and ESS accounting.",
            "family": "candidate_universe_expansion",
            "testability": "high",
            "pit_risk": "low",
            "tags": ["bnr", "sample-support", "lineage"],
        },
        {
            "prior_id": "DP-0003",
            "source": default_source,
            "topic": "followthrough",
            "claim": "Strong reclaims should express early follow-through; delayed continuation is a different trade archetype.",
            "measurable_translation": "Predict runner, scratch, chop, and late-reversal trade paths from the first few post-reclaim bars.",
            "family": "exit_behavior_research",
            "testability": "high",
            "pit_risk": "low",
            "tags": ["bnr", "trade-lifecycle", "followthrough"],
        },
        {
            "prior_id": "DP-0004",
            "source": default_source,
            "topic": "sequence_shape",
            "claim": "Static event features may miss the sequence shape of reclaim quality and failed continuation.",
            "measurable_translation": "Encode compact sequence descriptors for break, rejection, reclaim, and early post-entry path geometry.",
            "family": "feature",
            "testability": "medium",
            "pit_risk": "medium",
            "tags": ["bnr", "sequence-shape", "features"],
        },
        {
            "prior_id": "DP-0005",
            "source": default_source,
            "topic": "eligibility_filter",
            "claim": "BNR should only be eligible in environments that support continuation quality.",
            "measurable_translation": "Model environment-first eligibility before entry decisions and compare against unconditional BNR.",
            "family": "setup",
            "testability": "high",
            "pit_risk": "low",
            "tags": ["bnr", "eligibility", "environment"],
        },
    ]
    return priors


def build_hypotheses_from_priors(priors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    for index, prior in enumerate(priors, start=1):
        family = str(prior["family"])
        hypotheses.append(
            {
                "hypothesis_id": f"H-{index:05d}",
                "source": "domain_research_agent",
                "source_id": prior["prior_id"],
                "title": prior["topic"],
                "claim": prior["claim"],
                "measurable_translation": prior["measurable_translation"],
                "family": family,
                "status": "untested",
                "priority": _base_family_priority(family),
                "risk": "medium" if prior.get("pit_risk") == "medium" else "low",
                "budget": {"max_trials": _family_trial_budget(family), "max_cycles": 1},
                "dependencies": [],
                "blocked_by": [],
                "evidence_for": [],
                "evidence_against": [],
            }
        )
    return hypotheses


def build_research_backlog(
    hypotheses: list[dict[str, Any]],
    failure_memory: list[dict[str, Any]],
    *,
    stage2_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    stage2 = dict(stage2_result or {})
    sessions = int(stage2.get("data_quality", {}).get("sessions", 0) or 0)
    backlog: list[dict[str, Any]] = []
    failed_families = {str(item.get("family")) for item in failure_memory if item.get("status") in {"freeze", "reject", "failed_validation"}}
    failed_hypotheses = {
        str(item.get("hypothesis_id"))
        for item in failure_memory
        if item.get("status") in {"freeze", "reject", "failed_validation"} and item.get("hypothesis_id")
    }
    recent_failure_types = {str(item.get("failure_type")) for item in failure_memory if item.get("failure_type")}

    for row in hypotheses:
        item = dict(row)
        family = str(item.get("family", "unknown"))
        hypothesis_id = str(item.get("hypothesis_id", ""))
        score = float(item.get("priority", 0.5) or 0.5)
        blocked_by: list[str] = []
        reason_bits = [f"base priority {score:.2f}"]

        if hypothesis_id and hypothesis_id in failed_hypotheses:
            blocked_by.append("prior_failed_hypothesis")
            score -= 0.75
            reason_bits.append("blocked by prior failed hypothesis")
        if "validation_failed" in recent_failure_types and family in {"model", "threshold", "translation_policy"}:
            blocked_by.append("validation_failed_lineage")
            score -= 0.45
            reason_bits.append("blocked by failed validation lineage")
        if "cpcv_tail_path_fragility" in recent_failure_types:
            if family in {"candidate_universe_expansion", "setup", "exit_behavior_research", "subtype"}:
                score += 0.2
                reason_bits.append("boosted by CPCV tail failure memory")
            if family in {"model", "holdout_confirmation"}:
                blocked_by.append("cpcv_tail_path_fragility")
                score -= 0.35
                reason_bits.append("deprioritized until robustness improves")
        if "walk_forward_failure" in recent_failure_types and family in {"feature", "setup", "label"}:
            score += 0.15
            reason_bits.append("boosted by walk-forward failure memory")
        if family in failed_families and family not in {"setup", "feature"}:
            score -= 0.1
            reason_bits.append("family has prior failed branch")
        if sessions and sessions < 160 and family in {"model", "translation_policy"}:
            score -= 0.1
            reason_bits.append("sample floor not yet convincing for micro-tuning")

        item["priority_score"] = round(score, 4)
        item["blocked_by"] = blocked_by
        item["research_reason"] = "; ".join(reason_bits)
        backlog.append(item)

    backlog.sort(key=lambda item: (bool(item.get("blocked_by")), -float(item.get("priority_score", 0.0) or 0.0), str(item.get("hypothesis_id", ""))))
    return backlog


def build_research_director_summary(state: dict[str, Any]) -> dict[str, Any]:
    priors = list(state.get("domain_priors", []) or [])
    backlog = list(state.get("research_backlog", []) or [])
    failure_memory = list(state.get("failure_memory", []) or [])
    summary = {
        "status": "ready" if priors and backlog else "needs_domain_research",
        "domain_priors_loaded": bool(priors),
        "backlog_size": len(backlog),
        "recent_failure_count": len(failure_memory),
        "recommended_action": "research_domain_priors" if not priors else "rank_hypotheses",
        "top_hypothesis": dict(backlog[0]) if backlog else {},
    }
    if backlog:
        top = dict(backlog[0])
        summary["recommended_family"] = top.get("family")
        summary["why_now"] = top.get("research_reason")
    return summary


def build_research_director_plan(state: dict[str, Any], fallback_plan: dict[str, Any]) -> dict[str, Any]:
    backlog = list(state.get("research_backlog", []) or [])
    summary = dict(state.get("research_director_summary", {}) or {})
    plan = dict(fallback_plan)
    if not backlog:
        plan.setdefault("status", "ready")
        plan["next_action"] = "research_domain_priors"
        plan["assigned_research_action"] = "domain_prior_ingestion"
        plan["blocked_actions"] = ["holdout", "micro_tuning"]
        plan["reason"] = "Research backlog is empty; ingest domain priors before further iteration."
        plan["research_director"] = summary
        return plan

    selected_family = str(plan.get("selected_family") or plan.get("controller_override", {}).get("active_family") or "")
    hypothesis = _pick_hypothesis_for_family(backlog, selected_family) if selected_family else _top_viable_hypothesis(backlog)
    if not hypothesis:
        hypothesis = _top_viable_hypothesis(backlog) or dict(backlog[0])
        selected_family = str(hypothesis.get("family", selected_family))
        if selected_family and not plan.get("selected_family"):
            plan["selected_family"] = selected_family
            plan.setdefault("controller_override", {})
            plan["controller_override"]["active_family"] = selected_family
    elif hypothesis.get("blocked_by"):
        viable = _top_viable_hypothesis(backlog)
        if viable:
            hypothesis = viable
            selected_family = str(hypothesis.get("family", selected_family))
            plan["selected_family"] = selected_family
            plan.setdefault("controller_override", {})
            plan["controller_override"]["active_family"] = selected_family

    blocked_actions = _blocked_actions(state, hypothesis)
    assigned_action = _choose_research_action(state, hypothesis, selected_family)
    plan["next_action"] = "run_family_experiment" if summary.get("domain_priors_loaded") else "research_domain_priors"
    plan["assigned_research_action"] = assigned_action if assigned_action in available_research_actions() else None
    plan["hypothesis_id"] = hypothesis.get("hypothesis_id")
    plan["hypothesis_claim"] = hypothesis.get("claim")
    plan["measurable_translation"] = hypothesis.get("measurable_translation")
    plan["blocked_actions"] = blocked_actions
    plan["candidate_next_families"] = [
        {
            "family": row.get("family"),
            "hypothesis_id": row.get("hypothesis_id"),
            "priority_score": row.get("priority_score"),
            "blocked_by": row.get("blocked_by", []),
        }
        for row in backlog[:5]
    ]
    plan["research_director"] = {
        **summary,
        "active_hypothesis": hypothesis,
        "blocked_actions": blocked_actions,
        "assigned_research_action": assigned_action if assigned_action in available_research_actions() else None,
    }
    return plan


def build_failure_memory_entry(state: dict[str, Any]) -> dict[str, Any] | None:
    audit = dict(state.get("audit_summary", {}) or {})
    translation = dict(state.get("translation_summary", {}) or {})
    promotion_decision = str(state.get("promotion_decision", "") or "")
    next_step_plan = dict(state.get("next_step_plan", {}) or {})
    active_hypothesis = dict(state.get("active_hypothesis", {}) or {})

    failure_type = ""
    if dict(audit.get("cpcv", {}) or {}).get("status") == "fail":
        failure_type = "cpcv_tail_path_fragility"
    elif dict(audit.get("walk_forward", {}) or {}).get("status") == "fail":
        failure_type = "walk_forward_failure"
    elif translation.get("status") == "fail":
        failure_type = "translation_failure"
    elif promotion_decision == "reject":
        failure_type = "validation_failed"
    elif state.get("blocking_issues"):
        failure_type = "blocking_issue"

    if not failure_type:
        return None

    family = str(next_step_plan.get("selected_family") or active_hypothesis.get("family") or state.get("executed_research_family") or "unknown")
    return {
        "memory_id": f"FM-{utc_now_iso()}-{family}",
        "created_at": utc_now_iso(),
        "family": family,
        "hypothesis_id": active_hypothesis.get("hypothesis_id"),
        "failure_type": failure_type,
        "status": promotion_decision or "freeze",
        "reason": next_step_plan.get("reason") or translation.get("status") or failure_type,
        "evidence": {
            "cpcv_status": dict(audit.get("cpcv", {}) or {}).get("status"),
            "walk_forward_status": dict(audit.get("walk_forward", {}) or {}).get("status"),
            "translation_status": translation.get("status"),
            "blocking_issues": list(state.get("blocking_issues", []) or [])[:3],
        },
    }


def append_failure_memory(state: dict[str, Any]) -> list[dict[str, Any]]:
    existing = list(state.get("failure_memory", []) or [])
    entry = build_failure_memory_entry(state)
    if entry is None:
        return existing
    signature = (
        entry.get("family"),
        entry.get("hypothesis_id"),
        entry.get("failure_type"),
        entry.get("status"),
    )
    for row in existing:
        row_signature = (
            row.get("family"),
            row.get("hypothesis_id"),
            row.get("failure_type"),
            row.get("status"),
        )
        if row_signature == signature:
            return existing
    return [*existing, entry]


def _pick_hypothesis_for_family(backlog: list[dict[str, Any]], family: str) -> dict[str, Any]:
    for row in backlog:
        if str(row.get("family")) == family and not row.get("blocked_by"):
            return dict(row)
    for row in backlog:
        if str(row.get("family")) == family:
            return dict(row)
    return {}


def _top_viable_hypothesis(backlog: list[dict[str, Any]]) -> dict[str, Any]:
    for row in backlog:
        if not row.get("blocked_by"):
            return dict(row)
    return {}


def count_viable_hypotheses(backlog: list[dict[str, Any]]) -> int:
    return sum(1 for row in backlog if not row.get("blocked_by"))


def _choose_research_action(state: dict[str, Any], hypothesis: dict[str, Any], selected_family: str) -> str | None:
    priors_loaded = bool(state.get("domain_priors"))
    if not priors_loaded:
        return "domain_prior_ingestion"

    history = list(state.get("research_action_history", []) or [])
    failure_memory = list(state.get("failure_memory", []) or [])
    latest_failure = dict(failure_memory[-1]) if failure_memory else {}
    latest_failure_type = str(latest_failure.get("failure_type", "") or "")
    setup_failures = sum(1 for row in failure_memory if str(row.get("family")) == "setup")
    hypothesis_id = str(hypothesis.get("hypothesis_id", "") or "")

    def action_used(action_id: str) -> bool:
        for row in history:
            if str(row.get("action_id")) != action_id:
                continue
            if row.get("hypothesis_id") == hypothesis_id or row.get("family") == selected_family:
                return True
        return False

    if selected_family in {"candidate_universe_expansion", "exit_behavior_research", "feature"}:
        return selected_family if selected_family in available_research_actions() else None
    if latest_failure_type == "cpcv_tail_path_fragility" and not action_used("cpcv_attribution"):
        return "cpcv_attribution"
    if latest_failure_type and not action_used("validation_failure_analysis"):
        return "validation_failure_analysis"
    if selected_family == "setup" and setup_failures >= 2 and not action_used("setup_redesign"):
        return "setup_redesign"
    if selected_family == "setup" and setup_failures >= 1 and not action_used("candidate_universe_expansion"):
        return "candidate_universe_expansion"
    if selected_family == "setup" and setup_failures >= 1 and not action_used("exit_behavior_research"):
        return "exit_behavior_research"
    if selected_family == "setup" and action_used("setup_redesign") and not action_used("ml4t_backtest"):
        return "ml4t_backtest"
    if selected_family in available_research_actions():
        return selected_family
    return None


def _blocked_actions(state: dict[str, Any], hypothesis: dict[str, Any]) -> list[str]:
    blocked = set(hypothesis.get("blocked_by", []) or [])
    if any("holdout" in item.lower() for item in state.get("blocking_issues", [])):
        blocked.add("holdout")
    for row in state.get("failure_memory", []) or []:
        if row.get("failure_type") == "validation_failed":
            blocked.update({"holdout", "micro_tuning"})
    return sorted(blocked)


def _base_family_priority(family: str) -> float:
    priorities = {
        "setup": 0.82,
        "candidate_universe_expansion": 0.79,
        "exit_behavior_research": 0.76,
        "feature": 0.71,
        "subtype": 0.69,
        "label": 0.64,
        "translation_policy": 0.5,
        "model": 0.42,
    }
    return priorities.get(family, 0.45)


def _family_trial_budget(family: str) -> int:
    budgets = {
        "setup": 4,
        "candidate_universe_expansion": 4,
        "exit_behavior_research": 6,
        "feature": 4,
        "subtype": 4,
        "label": 4,
        "translation_policy": 3,
        "model": 2,
    }
    return budgets.get(family, 2)
