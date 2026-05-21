from __future__ import annotations

from dataclasses import asdict
from typing import Any

from trading_ml.env import load_runtime_env
from trading_ml.llm import create_chat_model, llm_enabled
from trading_ml.agent_nodes import (
    audit_agent_node,
    backtest_agent_node,
    bnr_research_agent_node,
    checkpoint_payload,
    cto_agent_node,
    data_steward_agent_node,
    diagnosis_agent_node,
    domain_research_agent_node,
    feature_agent_node,
    governor_agent_node,
    iteration_controller_node,
    labeling_agent_node,
    model_agent_node,
    program_director_node,
    promotion_decision_node,
    research_director_agent_node,
    search_controller_agent_node,
    setup_redesign_agent_node,
    strategy_intake_agent_node,
    translation_checkpoint_node,
)
from trading_ml.agent_workflow import build_agent_loop_state, build_loop_limits
from trading_ml.agent_state import AgentLoopState
from trading_ml.bnr_research_desk import (
    desk_data_steward_node,
    desk_director_node,
    desk_governor_node,
    desk_memory_update_node,
    eligibility_modeler_node,
    event_librarian_node,
    exit_research_agent_node,
    failure_analyst_node,
    feature_engineer_node,
    path_modeler_node,
    route_after_desk_director,
    setup_spec_agent_node,
)


def require_langgraph() -> tuple[Any, Any, Any, Any]:
    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Command, interrupt
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is not installed. Install the optional agents dependencies with "
            "`pip install -e .[agents]` before compiling the Stage 3 workflow."
        ) from exc
    return StateGraph, START, END, (InMemorySaver, Command, interrupt)


def _interrupt_for_review(state: AgentLoopState, checkpoint_name: str) -> dict[str, Any]:
    approvals = dict(state.get("approvals", {}))
    if approvals.get(checkpoint_name, False):
        pending = [name for name in state.get("checkpoints_pending", []) if name != checkpoint_name]
        return {"approvals": approvals, "checkpoints_pending": pending}
    _, _, _, (_, _, interrupt) = require_langgraph()
    decision = interrupt(asdict(checkpoint_payload(checkpoint_name, state)))
    approvals[checkpoint_name] = bool(decision)
    pending = [name for name in state.get("checkpoints_pending", []) if name != checkpoint_name]
    return {"approvals": approvals, "checkpoints_pending": pending}


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


def _llm_note(llm: Any, system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return _content_to_text(response)


def compile_bnr_langgraph(checkpointer: Any | None = None, llm: Any | None = None, *, use_llm: bool = True) -> Any:
    StateGraph, START, END, (InMemorySaver, Command, _) = require_langgraph()
    limits = build_loop_limits()
    load_runtime_env()
    llm_client = llm
    if llm_client is None and use_llm and llm_enabled():
        llm_client = create_chat_model(temperature=0)

    def strategy_intake(state: AgentLoopState) -> dict[str, Any]:
        return strategy_intake_agent_node(state)

    def research_director(state: AgentLoopState) -> dict[str, Any]:
        return research_director_agent_node(state)

    def domain_research(state: AgentLoopState) -> dict[str, Any]:
        return domain_research_agent_node(state)

    def program_director(state: AgentLoopState) -> dict[str, Any]:
        return program_director_node(state)

    def setup_redesign(state: AgentLoopState) -> dict[str, Any]:
        return setup_redesign_agent_node(state)

    def governor(state: AgentLoopState) -> dict[str, Any]:
        return governor_agent_node(state)

    def cto(state: AgentLoopState) -> dict[str, Any]:
        return cto_agent_node(state)

    def data_steward(state: AgentLoopState) -> dict[str, Any]:
        return data_steward_agent_node(state)

    def review_bnr_spec(state: AgentLoopState):
        update = _interrupt_for_review(state, "bnr_spec_approval")
        return Command(update=update, goto="bnr_research_agent")

    def bnr_research(state: AgentLoopState) -> dict[str, Any]:
        update = bnr_research_agent_node(state)
        if llm_client is not None:
            note = _llm_note(
                llm_client,
                "You are a trading ML research planner. Give concise next-step guidance.",
                (
                    "We are building a BNR 1m classification system around the 09:30:00-09:30:59 zone. "
                    f"Current BNR spec: {state.get('bnr_spec', {})}. "
                    "Return the next concrete research actions and what must be fixed before model training."
                ),
            )
            bnr_spec = dict(update.get("bnr_spec", {}))
            bnr_spec["llm_research_plan"] = note
            update["bnr_spec"] = bnr_spec
        return update

    def review_label(state: AgentLoopState):
        update = _interrupt_for_review(state, "label_approval")
        return Command(update=update, goto="feature_agent")

    def labeling(state: AgentLoopState) -> dict[str, Any]:
        return labeling_agent_node(state)

    def feature(state: AgentLoopState) -> dict[str, Any]:
        update = feature_agent_node(state)
        if llm_client is not None:
            note = _llm_note(
                llm_client,
                "You design leakage-safe trading features. Be concise and concrete.",
                (
                    "Suggest feature families for a BNR 1m setup classifier. "
                    "Use only information available by decision time. "
                    f"Current BNR spec: {state.get('bnr_spec', {})}."
                ),
            )
            feature_spec = dict(update.get("feature_spec", {}))
            feature_spec["llm_feature_ideas"] = note
            update["feature_spec"] = feature_spec
        return update

    def model_node(state: AgentLoopState) -> dict[str, Any]:
        update = model_agent_node(state)
        if llm_client is not None:
            note = _llm_note(
                llm_client,
                "You choose model plans for systematic trading research. Prefer conservative baselines first.",
                (
                    "Recommend the first model stack for a BNR take/don't-take classifier. "
                    "Assume Databento 30s and 1m data from 2024-2026 Mar, "
                    "and focus on calibrated, interpretable baselines before complexity."
                ),
            )
            model_spec = dict(update.get("model_spec", {}))
            model_spec["llm_model_plan"] = note
            update["model_spec"] = model_spec
        return update

    def backtest(state: AgentLoopState) -> dict[str, Any]:
        return backtest_agent_node(state)

    def review_search_space(state: AgentLoopState):
        update = _interrupt_for_review(state, "search_space_approval")
        return Command(update=update, goto="search_controller_agent")

    def search_controller(state: AgentLoopState) -> dict[str, Any]:
        return search_controller_agent_node(state, limits)

    def audit(state: AgentLoopState) -> dict[str, Any]:
        update = audit_agent_node(state)
        if llm_client is not None:
            note = _llm_note(
                llm_client,
                "You audit trading ML workflows for leakage, overfitting, and validation integrity.",
                (
                    "Given this workflow state, summarize the most likely audit failures and next checks. "
                    f"State snapshot: phase={state.get('phase')}, "
                    f"bnr_spec={state.get('bnr_spec', {})}, feature_spec={state.get('feature_spec', {})}, "
                    f"model_spec={state.get('model_spec', {})}."
                ),
            )
            audit_summary = dict(update.get("audit_summary", {}))
            audit_summary["llm_audit_focus"] = note
            update["audit_summary"] = audit_summary
        return update

    def translation(state: AgentLoopState) -> dict[str, Any]:
        return translation_checkpoint_node(state)

    def review_frozen_spec(state: AgentLoopState):
        update = _interrupt_for_review(state, "frozen_spec_approval")
        return Command(update=update, goto="diagnosis_agent")

    def diagnosis(state: AgentLoopState) -> dict[str, Any]:
        return diagnosis_agent_node(state)

    def promotion(state: AgentLoopState) -> dict[str, Any]:
        return promotion_decision_node(state)

    def iteration_controller(state: AgentLoopState) -> dict[str, Any]:
        return iteration_controller_node(state)

    def route_after_research_director(state: AgentLoopState) -> str:
        summary = dict(state.get("research_director_summary", {}) or {})
        if summary.get("recommended_action") == "research_domain_priors":
            return "domain_research_agent"
        return "program_director"

    def route_after_governor(state: AgentLoopState) -> str:
        if state.get("blocking_issues"):
            return "diagnosis_agent"
        if "bnr_spec_approval" in state.get("checkpoints_pending", []):
            return "review_bnr_spec"
        return "cto_agent"

    def route_after_program_director(state: AgentLoopState) -> str:
        plan = dict(state.get("next_step_plan", {}) or {})
        if plan.get("benchmark_status") == "exhausted_or_structurally_fragile":
            return "setup_redesign_agent"
        planned_family = dict(plan.get("controller_override", {}) or {}).get("active_family")
        executed_family = state.get("executed_research_family") or dict(state.get("search_results", {}) or {}).get("family")
        executed_cycle = int(state.get("executed_family_cycle", 0) or 0)
        current_cycle = int(state.get("research_cycle", 1) or 1)
        already_executed = (
            state.get("search_batch_status") == "complete"
            and executed_family == planned_family
            and executed_cycle == current_cycle
        )
        if already_executed:
            return "audit_agent"
        if plan.get("approval_required") == "search_space_approval" and planned_family and not already_executed and not state.get("translation_summary"):
            return "governor_agent"
        if state.get("translation_summary"):
            return "review_frozen_spec"
        return "governor_agent"

    def route_after_translation(state: AgentLoopState) -> str:
        if state.get("search_batch_status") == "complete":
            return "review_frozen_spec"
        return "program_director"

    graph = StateGraph(AgentLoopState)
    graph.add_node("strategy_intake_agent", strategy_intake)
    graph.add_node("research_director_agent", research_director)
    graph.add_node("domain_research_agent", domain_research)
    graph.add_node("program_director", program_director)
    graph.add_node("setup_redesign_agent", setup_redesign)
    graph.add_node("governor_agent", governor)
    graph.add_node("cto_agent", cto)
    graph.add_node("data_steward_agent", data_steward)
    graph.add_node("review_bnr_spec", review_bnr_spec)
    graph.add_node("bnr_research_agent", bnr_research)
    graph.add_node("labeling_agent", labeling)
    graph.add_node("review_label", review_label)
    graph.add_node("feature_agent", feature)
    graph.add_node("model_agent", model_node)
    graph.add_node("backtest_agent", backtest)
    graph.add_node("review_search_space", review_search_space)
    graph.add_node("search_controller_agent", search_controller)
    graph.add_node("audit_agent", audit)
    graph.add_node("translation_checkpoint", translation)
    graph.add_node("review_frozen_spec", review_frozen_spec)
    graph.add_node("diagnosis_agent", diagnosis)
    graph.add_node("promotion_decision", promotion)
    graph.add_node("iteration_controller", iteration_controller)

    graph.add_edge(START, "strategy_intake_agent")
    graph.add_edge("strategy_intake_agent", "research_director_agent")
    graph.add_conditional_edges("research_director_agent", route_after_research_director)
    graph.add_edge("domain_research_agent", "program_director")
    graph.add_conditional_edges("program_director", route_after_program_director)
    graph.add_edge("setup_redesign_agent", END)
    graph.add_conditional_edges("governor_agent", route_after_governor)
    graph.add_edge("cto_agent", "data_steward_agent")
    graph.add_edge("data_steward_agent", "bnr_research_agent")
    graph.add_edge("review_bnr_spec", "bnr_research_agent")
    graph.add_edge("bnr_research_agent", "labeling_agent")
    graph.add_edge("labeling_agent", "review_label")
    graph.add_edge("review_label", "feature_agent")
    graph.add_edge("feature_agent", "model_agent")
    graph.add_edge("model_agent", "backtest_agent")
    graph.add_edge("backtest_agent", "review_search_space")
    graph.add_edge("review_search_space", "search_controller_agent")
    graph.add_edge("search_controller_agent", "audit_agent")
    graph.add_edge("audit_agent", "translation_checkpoint")
    graph.add_conditional_edges("translation_checkpoint", route_after_translation)
    graph.add_edge("review_frozen_spec", "diagnosis_agent")
    graph.add_edge("diagnosis_agent", "promotion_decision")
    graph.add_edge("promotion_decision", "iteration_controller")

    def route_after_iteration(state: AgentLoopState) -> str:
        prior = list(state.get("run_log", []))
        if prior and prior[-1].get("actor") == "iteration_controller":
            payload = dict(prior[-1].get("payload", {}))
            if payload.get("continue_iteration"):
                return "research_director_agent"
        return END

    graph.add_conditional_edges("iteration_controller", route_after_iteration)

    saver = checkpointer or InMemorySaver()
    return graph.compile(checkpointer=saver)


def compile_bnr_research_desk_graph(checkpointer: Any | None = None) -> Any:
    StateGraph, START, END, (InMemorySaver, _, _) = require_langgraph()
    load_runtime_env()

    graph = StateGraph(AgentLoopState)
    graph.add_node("desk_data_steward", desk_data_steward_node)
    graph.add_node("event_librarian", event_librarian_node)
    graph.add_node("failure_analyst", failure_analyst_node)
    graph.add_node("desk_director", desk_director_node)
    graph.add_node("feature_engineer", feature_engineer_node)
    graph.add_node("setup_spec_agent", setup_spec_agent_node)
    graph.add_node("eligibility_modeler", eligibility_modeler_node)
    graph.add_node("path_modeler", path_modeler_node)
    graph.add_node("exit_research_agent", exit_research_agent_node)
    graph.add_node("desk_governor", desk_governor_node)
    graph.add_node("desk_memory_update", desk_memory_update_node)

    graph.add_edge(START, "desk_data_steward")
    graph.add_edge("desk_data_steward", "event_librarian")
    graph.add_edge("event_librarian", "failure_analyst")
    graph.add_edge("failure_analyst", "desk_director")
    graph.add_conditional_edges("desk_director", route_after_desk_director)
    graph.add_edge("feature_engineer", "desk_governor")
    graph.add_edge("setup_spec_agent", "desk_governor")
    graph.add_edge("eligibility_modeler", "desk_governor")
    graph.add_edge("path_modeler", "desk_governor")
    graph.add_edge("exit_research_agent", "desk_governor")
    graph.add_edge("desk_governor", "desk_memory_update")
    graph.add_edge("desk_memory_update", END)

    saver = checkpointer or InMemorySaver()
    return graph.compile(checkpointer=saver)


def build_langgraph_initial_input(
    *,
    preapproved_checkpoints: list[str] | None = None,
    max_research_cycles: int | None = None,
    compute_budget_overrides: dict[str, Any] | None = None,
    runtime_profile: str = "standard",
) -> AgentLoopState:
    return build_agent_loop_state(
        preapproved_checkpoints=preapproved_checkpoints,
        max_research_cycles=max_research_cycles,
        compute_budget_overrides=compute_budget_overrides,
        runtime_profile=runtime_profile,
    )


def build_bnr_research_desk_initial_input() -> AgentLoopState:
    return build_agent_loop_state(
        preapproved_checkpoints=["bnr_spec_approval", "label_approval", "search_space_approval", "frozen_spec_approval"],
        max_research_cycles=1,
        compute_budget_overrides={
            "max_trials": 1,
            "max_full_validations": 0,
            "max_cpcv_runs": 0,
            "max_model_trains": 1,
        },
        runtime_profile="bounded_autonomous",
    )


def build_governor_state_from_desk_handoff(desk_state: dict[str, Any]) -> AgentLoopState:
    state = build_agent_loop_state()
    state["stage2_result"] = dict(desk_state.get("stage2_result", {}) or {})
    state["bnr_attempts"] = list(desk_state.get("bnr_attempts", []) or [])
    state["failure_clusters"] = list(desk_state.get("failure_clusters", []) or [])
    state["desk_summary"] = dict(desk_state.get("desk_summary", {}) or {})
    state["desk_proposals"] = list(desk_state.get("desk_proposals", []) or [])
    state["desk_memory"] = list(desk_state.get("desk_memory", []) or [])
    state["run_log"] = list(desk_state.get("run_log", []) or [])
    return state
