from __future__ import annotations

from typing import Any

from trading_ml.agent_nodes import data_steward_agent_node
from trading_ml.artifact_store import persist_node_artifact
from trading_ml.bnr_attempts import build_bnr_attempts
from trading_ml.failure_clusters import build_failure_clusters
from trading_ml.research_memory_store import append_desk_memory_entry
from trading_ml.schemas import utc_now_iso


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
    return [*list(state.get("run_log", []) or []), entry]


def desk_data_steward_node(state: dict[str, Any]) -> dict[str, Any]:
    update = data_steward_agent_node(state)
    update["current_node"] = "desk_data_steward"
    return update


def event_librarian_node(state: dict[str, Any]) -> dict[str, Any]:
    stage2 = dict(state.get("stage2_result", {}) or {})
    audit = dict(state.get("audit_summary", {}) or {})
    walk_forward = dict(audit.get("walk_forward", {}) or {})
    prediction_records = list(walk_forward.get("stitched_prediction_records", []) or [])
    if not prediction_records:
        prediction_records = list(dict(stage2.get("model_summary", {}) or {}).get("prediction_records", []) or [])
    attempts = build_bnr_attempts(stage2, prediction_records)
    summary = {
        "status": "ready" if attempts else "pending",
        "attempt_count": len(attempts),
        "executed_count": sum(1 for row in attempts if row.get("executed")),
        "path_classes": _count_values(attempts, "path_class"),
        "subtypes": _count_values(attempts, "setup_subtype"),
    }
    return {
        "current_node": "event_librarian",
        "bnr_attempts": attempts,
        "desk_summary": {
            **dict(state.get("desk_summary", {}) or {}),
            "event_librarian": summary,
        },
        "run_log": _append_log(state, "event_librarian", "Stored BNR attempts with decision-time lineage.", summary),
    }


def failure_analyst_node(state: dict[str, Any]) -> dict[str, Any]:
    attempts = list(state.get("bnr_attempts", []) or [])
    clusters = build_failure_clusters(attempts)
    summary = {
        "status": "ready" if clusters else "pending",
        "cluster_count": len(clusters),
        "top_cluster": dict(clusters[0]) if clusters else {},
    }
    return {
        "current_node": "failure_analyst",
        "failure_clusters": clusters,
        "desk_summary": {
            **dict(state.get("desk_summary", {}) or {}),
            "failure_analyst": summary,
        },
        "run_log": _append_log(state, "failure_analyst", "Clustered repeated BNR failure shapes.", summary),
    }


def desk_director_node(state: dict[str, Any]) -> dict[str, Any]:
    clusters = list(state.get("failure_clusters", []) or [])
    top_cluster = dict(clusters[0]) if clusters else {}
    selected_node = _select_desk_node(top_cluster, state)
    summary = {
        "status": "ready",
        "selected_node": selected_node,
        "top_cluster": top_cluster,
        "reason": _selection_reason(top_cluster, selected_node),
    }
    return {
        "current_node": "desk_director",
        "desk_summary": {
            **dict(state.get("desk_summary", {}) or {}),
            "desk_director": summary,
        },
        "run_log": _append_log(state, "desk_director", "Selected the next BNR desk specialist from failure evidence.", summary),
    }


def feature_engineer_node(state: dict[str, Any]) -> dict[str, Any]:
    top_cluster = dict((state.get("failure_clusters", []) or [{}])[0])
    proposal = {
        "proposal_id": f"DPROP-{utc_now_iso()}-feature",
        "node": "feature_engineer",
        "family": "feature",
        "claim": f"Engineer features to separate {top_cluster.get('family', 'unknown')} attempts before entry.",
        "proposed_features": list(top_cluster.get("recommended_focus", []) or ["confirmation_strength", "auction_clarity"]),
        "derived_from_cluster": top_cluster.get("cluster_id"),
    }
    return _proposal_update(state, "feature_engineer", proposal, "Proposed feature work from clustered failure evidence.")


def setup_spec_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    top_cluster = dict((state.get("failure_clusters", []) or [{}])[0])
    proposal = {
        "proposal_id": f"DPROP-{utc_now_iso()}-setup",
        "node": "setup_spec_agent",
        "family": "setup",
        "claim": f"Refine BNR setup tolerances to reduce {top_cluster.get('family', 'unknown')}.",
        "parameter_knobs": [
            "max_retrace_depth",
            "min_confirmation_close_strength",
            "max_reclaim_failures",
        ],
        "derived_from_cluster": top_cluster.get("cluster_id"),
    }
    return _proposal_update(state, "setup_spec_agent", proposal, "Proposed parameterized BNR setup adjustments.")


def eligibility_modeler_node(state: dict[str, Any]) -> dict[str, Any]:
    attempts = list(state.get("bnr_attempts", []) or [])
    proposal = {
        "proposal_id": f"DPROP-{utc_now_iso()}-eligibility",
        "node": "eligibility_modeler",
        "family": "eligibility",
        "claim": "Model tradeable vs non-tradeable BNR attempts before entry.",
        "objective": "trade_no_trade",
        "candidate_state_axes": ["setup_subtype", "time_bucket", "probability_bucket", "path_class"],
        "sample_size": len(attempts),
    }
    return _proposal_update(state, "eligibility_modeler", proposal, "Proposed tradeability modeling from BNR attempts.")


def path_modeler_node(state: dict[str, Any]) -> dict[str, Any]:
    attempts = list(state.get("bnr_attempts", []) or [])
    proposal = {
        "proposal_id": f"DPROP-{utc_now_iso()}-path",
        "node": "path_modeler",
        "family": "path_modeling",
        "claim": "Model post-confirmation path class before exit design.",
        "path_classes": sorted({str(row.get('path_class', 'unknown')) for row in attempts}),
        "objective": "runner_vs_chop_vs_failure",
        "sample_size": len(attempts),
    }
    return _proposal_update(state, "path_modeler", proposal, "Proposed path-class modeling for BNR attempts.")


def exit_research_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    top_cluster = dict((state.get("failure_clusters", []) or [{}])[0])
    proposal = {
        "proposal_id": f"DPROP-{utc_now_iso()}-exit",
        "node": "exit_research_agent",
        "family": "exit_behavior_research",
        "claim": f"Map {top_cluster.get('family', 'unknown')} path failures into exit-policy tests.",
        "exit_families": ["scratch_timing", "partial_then_trail", "time_stop"],
        "derived_from_cluster": top_cluster.get("cluster_id"),
    }
    return _proposal_update(state, "exit_research_agent", proposal, "Proposed exit research from failure clusters.")


def desk_governor_node(state: dict[str, Any]) -> dict[str, Any]:
    proposals = list(state.get("desk_proposals", []) or [])
    latest = dict(proposals[-1]) if proposals else {}
    blocked_reasons: list[str] = []
    raw_text = " ".join(str(value) for value in latest.values())
    if "holdout" in raw_text.lower():
        blocked_reasons.append("holdout_reference_not_allowed")
    if "future" in raw_text.lower():
        blocked_reasons.append("future_reference_not_allowed")
    summary = {
        "status": "pass" if not blocked_reasons else "fail",
        "blocked_reasons": blocked_reasons,
        "proposal_id": latest.get("proposal_id"),
    }
    return {
        "current_node": "desk_governor",
        "desk_summary": {
            **dict(state.get("desk_summary", {}) or {}),
            "desk_governor": summary,
        },
        "run_log": _append_log(state, "desk_governor", "Checked desk proposal for undisciplined tuning or boundary violations.", summary),
    }


def desk_memory_update_node(state: dict[str, Any]) -> dict[str, Any]:
    proposals = list(state.get("desk_proposals", []) or [])
    latest = dict(proposals[-1]) if proposals else {}
    clusters = list(state.get("failure_clusters", []) or [])
    memory = [
        *list(state.get("desk_memory", []) or []),
        {
            "created_at": utc_now_iso(),
            "top_cluster": dict(clusters[0]) if clusters else {},
            "proposal": latest,
        },
    ]
    handoff = {
        "status": "ready_for_governor_graph" if latest else "pending",
        "proposal_id": latest.get("proposal_id"),
        "proposal_family": latest.get("family"),
        "top_cluster_family": dict(clusters[0]).get("family") if clusters else None,
    }
    if latest:
        append_desk_memory_entry(
            {
                "created_at": utc_now_iso(),
                "proposal_id": latest.get("proposal_id"),
                "proposal_family": latest.get("family"),
                "top_cluster_family": handoff.get("top_cluster_family"),
                "proposal": latest,
            }
        )
    return {
        "current_node": "desk_memory_update",
        "desk_memory": memory,
        "desk_summary": {
            **dict(state.get("desk_summary", {}) or {}),
            "desk_memory_update": handoff,
        },
        "run_log": _append_log(state, "desk_memory_update", "Stored desk proposal and prepared handoff to the governor graph.", handoff),
    }


def route_after_desk_director(state: dict[str, Any]) -> str:
    summary = dict(dict(state.get("desk_summary", {}) or {}).get("desk_director", {}) or {})
    return str(summary.get("selected_node", "feature_engineer"))


def _proposal_update(state: dict[str, Any], actor: str, proposal: dict[str, Any], message: str) -> dict[str, Any]:
    proposals = [*list(state.get("desk_proposals", []) or []), proposal]
    return {
        "current_node": actor,
        "desk_proposals": proposals,
        "desk_summary": {
            **dict(state.get("desk_summary", {}) or {}),
            actor: proposal,
        },
        "run_log": _append_log(state, actor, message, proposal),
    }


def _select_desk_node(top_cluster: dict[str, Any], state: dict[str, Any]) -> str:
    family = str(top_cluster.get("family", "") or "")
    recommended_family = str(top_cluster.get("recommended_family", "") or "")
    proposal_history = {
        str(row.get("proposal_family"))
        for row in list(state.get("desk_memory", []) or [])
        if row.get("proposal_family")
    }
    candidates = _candidate_proposal_families(family, recommended_family)
    for proposal_family in candidates:
        if proposal_family not in proposal_history:
            return _node_for_proposal_family(proposal_family)
    return _node_for_proposal_family(candidates[0] if candidates else "feature")


def _selection_reason(top_cluster: dict[str, Any], selected_node: str) -> str:
    if not top_cluster:
        return "No clustered failure evidence yet; default to feature engineering."
    return f"Top failure cluster {top_cluster.get('family')} suggests {selected_node}."


def _candidate_proposal_families(failure_family: str, recommended_family: str) -> list[str]:
    if failure_family == "no_follow_through":
        return ["path_modeling", "exit_behavior_research", "feature"]
    if failure_family == "deep_retrace_failure":
        return ["setup", "feature", "eligibility"]
    if failure_family == "no_reclaim_edge":
        return ["eligibility", "setup", "feature"]
    if failure_family == "weak_continuation":
        return ["feature", "path_modeling", "exit_behavior_research"]
    mapping = {
        "exit_behavior_research": ["exit_behavior_research", "path_modeling", "feature"],
        "setup": ["setup", "feature", "eligibility"],
        "feature": ["feature", "path_modeling", "setup"],
        "candidate_universe_expansion": ["eligibility", "setup", "feature"],
    }
    return mapping.get(recommended_family, ["feature", "setup"])


def _node_for_proposal_family(proposal_family: str) -> str:
    mapping = {
        "feature": "feature_engineer",
        "setup": "setup_spec_agent",
        "eligibility": "eligibility_modeler",
        "path_modeling": "path_modeler",
        "exit_behavior_research": "exit_research_agent",
    }
    return mapping.get(proposal_family, "feature_engineer")


def _count_values(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return [{"value": value, "count": count} for value, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)]
