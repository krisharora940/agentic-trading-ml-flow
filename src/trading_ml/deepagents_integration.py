from __future__ import annotations

from typing import Any

from trading_ml.env import load_runtime_env
from trading_ml.llm import get_default_model_name

TRADING_SUPERVISOR_PROMPT = """
You supervise a BNR 1m trading ML workflow.
Delegate work to specialized subagents.
Preserve the evidence boundary.
Do not allow holdout contamination.
Treat ML4T skills as procedural constraints, not optional guidance.
Escalate unresolved process risks to the governor and unresolved technical parity risks to the CTO.
Return concise structured summaries after each subtask.
""".strip()


def require_deepagents() -> Any:
    try:
        from deepagents import CompiledSubAgent, create_deep_agent
    except ImportError as exc:
        raise RuntimeError(
            "DeepAgents is not installed. Install the optional agents dependencies with "
            "`pip install -e .[agents]` before creating the Stage 3 supervisor."
        ) from exc
    return create_deep_agent, CompiledSubAgent


def create_bnr_supervisor(model: str, compiled_langgraph: Any) -> Any:
    create_deep_agent, CompiledSubAgent = require_deepagents()
    load_runtime_env()

    subagents = [
        CompiledSubAgent(
            name="governor",
            description="Enforces process discipline, phase gates, and evidence boundaries.",
            runnable=compiled_langgraph,
        ),
        {
            "name": "cto",
            "description": "Owns architecture, parity, safety checks, and operational readiness.",
            "system_prompt": "You are the technical second-in-command for a trading ML system.",
            "skills": ["trading-ml-cto"],
        },
        {
            "name": "data-steward",
            "description": "Validates Databento manifests, timestamps, and data quality assumptions.",
            "system_prompt": "Focus on timestamp integrity, manifests, and decision-time correctness.",
            "skills": ["ml4t-databento", "ml4t-point-in-time", "ml4t-data-leakage"],
        },
        {
            "name": "research",
            "description": "Defines BNR setup logic, candidate generation, and rule hypotheses.",
            "system_prompt": "Work on BNR setup definition and candidate structure only.",
            "skills": ["trading-ml-governor"],
        },
    ]

    return create_deep_agent(
        model=model or get_default_model_name(),
        system_prompt=TRADING_SUPERVISOR_PROMPT,
        subagents=subagents,
        name="bnr-supervisor",
    )
