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
    feature_agent_node,
    governor_agent_node,
    iteration_controller_node,
    labeling_agent_node,
    model_agent_node,
    promotion_decision_node,
    search_controller_agent_node,
    translation_checkpoint_node,
)
from trading_ml.agent_workflow import build_agent_loop_state, build_loop_limits
from trading_ml.agent_state import AgentLoopState


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

    def route_after_governor(state: AgentLoopState) -> str:
        if "bnr_spec_approval" in state.get("checkpoints_pending", []):
            return "review_bnr_spec"
        return "cto_agent"

    graph = StateGraph(AgentLoopState)
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

    graph.add_edge(START, "governor_agent")
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
    graph.add_edge("translation_checkpoint", "review_frozen_spec")
    graph.add_edge("review_frozen_spec", "diagnosis_agent")
    graph.add_edge("diagnosis_agent", "promotion_decision")
    graph.add_edge("promotion_decision", "iteration_controller")

    def route_after_iteration(state: AgentLoopState) -> str:
        prior = list(state.get("run_log", []))
        if prior and prior[-1].get("actor") == "iteration_controller":
            payload = dict(prior[-1].get("payload", {}))
            if payload.get("continue_iteration"):
                return "data_steward_agent"
        return END

    graph.add_conditional_edges("iteration_controller", route_after_iteration)

    saver = checkpointer or InMemorySaver()
    return graph.compile(checkpointer=saver)


def build_langgraph_initial_input() -> AgentLoopState:
    return build_agent_loop_state()
