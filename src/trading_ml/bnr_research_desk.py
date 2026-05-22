from __future__ import annotations

from typing import Any

from trading_ml.agent_nodes import data_steward_agent_node
from trading_ml.artifact_store import persist_node_artifact
from trading_ml.bnr_attempts import build_bnr_attempts
from trading_ml.failure_clusters import build_failure_clusters
from trading_ml.price_action_feature_catalog import build_catalog_feature_proposals
from trading_ml.research_memory_store import append_desk_memory_entry
from trading_ml.schemas import utc_now_iso


MAX_SAME_FAMILY_CYCLES = 2
MAX_FEATURE_CYCLES_WITHOUT_CPCV_IMPROVEMENT = 2
ALLOWED_PROPOSAL_FAMILIES = {
    "feature",
    "setup",
    "eligibility",
    "path_modeling",
    "exit_behavior_research",
}
ALLOWED_DESK_NODES = {
    "feature_engineer",
    "setup_spec_agent",
    "eligibility_modeler",
    "path_modeler",
    "exit_research_agent",
}


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
        "setup_states": _count_values(attempts, "setup_state"),
        "environment_states": _count_values(attempts, "environment_state"),
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


def price_action_expert_node(state: dict[str, Any], llm: Any | None = None) -> dict[str, Any]:
    top_cluster = dict((state.get("failure_clusters", []) or [{}])[0])
    attempts = list(state.get("bnr_attempts", []) or [])
    target_setup_state = str(top_cluster.get("dominant_setup_state", "") or "unknown")
    target_environment_state = str(top_cluster.get("dominant_environment_state", "") or "unknown")
    target_path_class = str(dict(top_cluster.get("evidence", {}) or {}).get("path_class_mode", "") or "unknown")
    fallback = _heuristic_price_action_expert(top_cluster)
    if llm is not None:
        try:
            note = llm.bind(max_tokens=220).invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a concise institutional price action researcher. "
                            "Given BNR failure evidence, propose one bounded next research direction. "
                            "Return plain JSON with keys: recommended_family, recommended_node, hypothesis, "
                            "proposed_feature_concepts, proposed_parameter_knobs, exit_focus, kill_criteria. "
                            "Be brief and keep the JSON compact."
                        ),
                    },
                    {
                        "role": "user",
                        "content": str(
                            {
                                "failure_family": top_cluster.get("family"),
                                "setup_state": target_setup_state,
                                "environment_state": target_environment_state,
                                "path_class": target_path_class,
                                "recommended_focus": top_cluster.get("recommended_focus", []),
                                "rows": top_cluster.get("rows"),
                                "attempt_count": len(attempts),
                            }
                        ),
                    },
                ]
            )
            fallback["llm_note"] = _content_to_text(note)
            parsed = _parse_expert_note(fallback["llm_note"])
            if parsed:
                fallback.update(_sanitize_expert_note(parsed, fallback))
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            fallback["llm_error"] = str(exc)
    summary = {
        **fallback,
        "target_setup_state": target_setup_state,
        "target_environment_state": target_environment_state,
        "target_path_class": target_path_class,
        "status": "ready",
    }
    return {
        "current_node": "price_action_expert",
        "price_action_expert": summary,
        "desk_summary": {
            **dict(state.get("desk_summary", {}) or {}),
            "price_action_expert": summary,
        },
        "run_log": _append_log(state, "price_action_expert", "Proposed bounded price-action hypotheses from clustered BNR failure evidence.", summary),
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
    feature_plan = build_catalog_feature_proposals(
        str(state.get("strategy_notes", "") or ""),
        top_cluster=top_cluster,
        bnr_spec=dict(state.get("bnr_spec", {}) or {}),
        limit=6,
    )
    proposal = {
        "proposal_id": f"DPROP-{utc_now_iso()}-feature",
        "node": "feature_engineer",
        "family": "feature",
        "claim": feature_plan["feature_claim"],
        "proposed_features": [row["feature_name"] for row in feature_plan.get("feature_catalog_candidates", [])],
        "feature_catalog_candidates": feature_plan.get("feature_catalog_candidates", []),
        "feature_catalog_groups": feature_plan.get("selected_feature_groups", []),
        "feature_labs": feature_plan.get("next_feature_labs", []),
        "research_questions": feature_plan.get("research_questions", []),
        "feature_catalog_version": feature_plan.get("catalog_version", 1),
        "target_failure_cluster": top_cluster.get("cluster_id"),
        "target_setup_state": top_cluster.get("dominant_setup_state"),
        "target_environment_state": top_cluster.get("dominant_environment_state"),
        "target_path_class": dict(top_cluster.get("evidence", {}) or {}).get("path_class_mode"),
        "expected_target_metrics": ["cpcv_pbo", "deflated_sharpe", "worst_path_loss", "calibration"],
        "kill_criteria": [
            "kill after two accepted feature cycles without CPCV or DSR improvement",
            "kill if the same catalog group repeats without a new target failure cluster",
            "kill if accepted feature lift only improves local walk-forward metrics",
        ],
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
        "target_setup_state": top_cluster.get("dominant_setup_state"),
        "target_environment_state": top_cluster.get("dominant_environment_state"),
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
        "candidate_state_axes": ["setup_state", "environment_state", "time_bucket", "probability_bucket"],
        "setup_states": sorted({str(row.get('setup_state', 'unknown')) for row in attempts}),
        "environment_states": sorted({str(row.get('environment_state', 'unknown')) for row in attempts}),
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
        "setup_states": sorted({str(row.get('setup_state', 'unknown')) for row in attempts}),
        "environment_states": sorted({str(row.get('environment_state', 'unknown')) for row in attempts}),
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
        "target_setup_state": top_cluster.get("dominant_setup_state"),
        "target_environment_state": top_cluster.get("dominant_environment_state"),
        "target_path_class": dict(top_cluster.get("evidence", {}) or {}).get("path_class_mode"),
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
            "proposal_id": latest.get("proposal_id"),
            "proposal_family": latest.get("family"),
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
    expert = dict(state.get("price_action_expert", {}) or {})
    proposal_history = {
        str(row.get("proposal_family"))
        for row in list(state.get("desk_memory", []) or [])
        if row.get("proposal_family")
    }
    governed_records = list(state.get("research_action_history", []) or [])
    governed_history = {str(row.get("family")) for row in governed_records if row.get("family")}
    preferred_node = str(expert.get("recommended_node", "") or "")
    preferred_family = str(expert.get("recommended_family", "") or "")
    if preferred_node and preferred_family:
        if (
            preferred_family not in proposal_history
            and not _proposal_family_already_executed(preferred_family, governed_history)
            and not _proposal_family_loop_blocked(preferred_family, governed_records)
        ):
            return preferred_node
    candidates = _candidate_proposal_families(family, recommended_family)
    for proposal_family in candidates:
        if (
            proposal_family not in proposal_history
            and not _proposal_family_already_executed(proposal_family, governed_history)
            and not _proposal_family_loop_blocked(proposal_family, governed_records)
        ):
            return _node_for_proposal_family(proposal_family)
    ranked_fallbacks = sorted(
        candidates or ["feature"],
        key=lambda proposal_family: (
            _proposal_family_loop_blocked(proposal_family, governed_records),
            _proposal_family_already_executed(proposal_family, governed_history),
            _proposal_family_last_seen_index(proposal_family, list(state.get("desk_memory", []) or [])),
        ),
    )
    return _node_for_proposal_family(ranked_fallbacks[0])


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


def _proposal_family_already_executed(proposal_family: str, governed_history: set[str]) -> bool:
    mapping = {
        "eligibility": "candidate_universe_expansion",
        "setup": "setup",
        "feature": "feature",
        "path_modeling": "exit_behavior_research",
        "exit_behavior_research": "exit_behavior_research",
    }
    governed_family = mapping.get(proposal_family)
    return bool(governed_family and governed_family in governed_history)


def _proposal_family_loop_blocked(proposal_family: str, governed_records: list[dict[str, Any]]) -> bool:
    governed_family = {
        "eligibility": "candidate_universe_expansion",
        "setup": "setup",
        "feature": "feature",
        "path_modeling": "exit_behavior_research",
        "exit_behavior_research": "exit_behavior_research",
    }.get(proposal_family)
    if not governed_family:
        return False
    same_family_streak = 0
    for row in reversed(governed_records):
        if str(row.get("family")) == governed_family:
            same_family_streak += 1
            continue
        break
    if same_family_streak >= MAX_SAME_FAMILY_CYCLES:
        return True
    if governed_family != "feature":
        return False
    accepted = [
        row for row in governed_records
        if str(row.get("family")) == "feature" and str(row.get("batch_decision")) == "accept"
    ]
    if len(accepted) < MAX_FEATURE_CYCLES_WITHOUT_CPCV_IMPROVEMENT:
        return False
    recent = accepted[-MAX_FEATURE_CYCLES_WITHOUT_CPCV_IMPROVEMENT:]
    return not any(_has_robustness_improvement(row) for row in recent)


def _has_robustness_improvement(row: dict[str, Any]) -> bool:
    evidence = dict(row.get("marginal_evidence", {}) or {})
    return bool(
        evidence.get("cpcv_delta")
        or evidence.get("dsr_delta")
        or evidence.get("calibration_delta")
        or evidence.get("worst_path_loss_delta")
    )


def _proposal_family_last_seen_index(proposal_family: str, desk_memory: list[dict[str, Any]]) -> int:
    for idx in range(len(desk_memory) - 1, -1, -1):
        if str(desk_memory[idx].get("proposal_family")) == proposal_family:
            return len(desk_memory) - idx
    return 10_000


def _count_values(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return [{"value": value, "count": count} for value, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)]


def _heuristic_price_action_expert(top_cluster: dict[str, Any]) -> dict[str, Any]:
    family = str(top_cluster.get("family", "") or "")
    setup_state = str(top_cluster.get("dominant_setup_state", "") or "unknown")
    environment_state = str(top_cluster.get("dominant_environment_state", "") or "unknown")
    path_class = str(dict(top_cluster.get("evidence", {}) or {}).get("path_class_mode", "") or "unknown")
    if family == "no_follow_through":
        return {
            "recommended_family": "path_modeling",
            "recommended_node": "path_modeler",
            "hypothesis": f"{setup_state} setups in {environment_state} are path-class problems, not entry-geometry problems.",
            "proposed_feature_concepts": ["followthrough_strength", "opening_drive_strength", "candle_tempo_decay"],
            "proposed_parameter_knobs": ["scratch_timing", "trail_activation"],
            "exit_focus": ["runner_vs_chop", "late_reversal"],
            "kill_criteria": ["reject if focused path classes do not improve CPCV tail behavior"],
        }
    if family == "no_reclaim_edge":
        return {
            "recommended_family": "setup",
            "recommended_node": "setup_spec_agent",
            "hypothesis": f"{setup_state} attempts in {environment_state} need tighter reclaim confirmation rather than broader candidate expansion.",
            "proposed_feature_concepts": ["reclaim_body_strength", "reclaim_close_location", "opening_drive_strength"],
            "proposed_parameter_knobs": ["min_confirmation_close_strength", "max_retrace_depth", "max_reclaim_failures"],
            "exit_focus": [],
            "kill_criteria": ["reject if setup tightening reduces candidates without improving worst-path loss"],
        }
    if family in {"deep_retrace_failure", "weak_continuation"}:
        return {
            "recommended_family": "feature",
            "recommended_node": "feature_engineer",
            "hypothesis": f"{setup_state} setups in {environment_state} are missing state-quality features before entry.",
            "proposed_feature_concepts": ["followthrough_strength", "body_to_range_ratio", "distance_to_recent_high"],
            "proposed_parameter_knobs": ["min_confirmation_close_strength"],
            "exit_focus": ["fake_breakout_vs_repair"] if path_class == "failure" else [],
            "kill_criteria": ["reject after two accepted feature cycles without CPCV or calibration improvement"],
        }
    return {
        "recommended_family": "feature",
        "recommended_node": "feature_engineer",
        "hypothesis": "Current BNR state cluster needs more explicit price-action state features before further search.",
        "proposed_feature_concepts": ["followthrough_strength", "opening_drive_strength"],
        "proposed_parameter_knobs": [],
        "exit_focus": [],
        "kill_criteria": ["reject if no robustness lift after bounded feature cycle"],
    }


def _content_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return str(content)


def _parse_expert_note(note: str) -> dict[str, Any]:
    import json

    try:
        parsed = json.loads(note)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    allowed = {
        "recommended_family",
        "recommended_node",
        "hypothesis",
        "proposed_feature_concepts",
        "proposed_parameter_knobs",
        "exit_focus",
        "kill_criteria",
    }
    return {key: value for key, value in parsed.items() if key in allowed}


def _sanitize_expert_note(parsed: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(parsed)
    family = str(sanitized.get("recommended_family", "") or "").strip()
    node = str(sanitized.get("recommended_node", "") or "").strip()
    if family not in ALLOWED_PROPOSAL_FAMILIES:
        family = str(fallback.get("recommended_family", "feature") or "feature")
    if node not in ALLOWED_DESK_NODES:
        node = _node_for_proposal_family(family)
    sanitized["recommended_family"] = family
    sanitized["recommended_node"] = node
    return sanitized
