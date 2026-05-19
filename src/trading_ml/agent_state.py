from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from trading_ml.schemas import EvidenceBoundary


CheckpointName = Literal[
    "bnr_spec_approval",
    "label_approval",
    "search_space_approval",
    "frozen_spec_approval",
]

DecisionName = Literal["reject", "revise", "freeze", "advance_to_validation"]

FailureCategory = Literal[
    "data_issue",
    "feature_issue",
    "label_issue",
    "model_issue",
    "execution_issue",
    "unknown",
]


class AgentLoopState(TypedDict, total=False):
    program_state: dict[str, Any]
    next_step_plan: dict[str, Any]
    strategy_notes: str
    research_intake: dict[str, Any]
    phase: str
    current_node: str
    evidence_boundary: dict[str, Any]
    bnr_spec: dict[str, Any]
    label_spec: dict[str, Any]
    feature_spec: dict[str, Any]
    feature_diagnostics: dict[str, Any]
    model_spec: dict[str, Any]
    stage2_config: dict[str, Any]
    stage2_result: dict[str, Any]
    controller_state: dict[str, Any]
    search_space: dict[str, Any]
    search_results: dict[str, Any]
    translation_summary: dict[str, Any]
    frozen_benchmark: dict[str, Any]
    approvals: dict[str, bool]
    checkpoints_pending: list[str]
    experiment_counts: dict[str, int]
    research_cycle: int
    max_research_cycles: int
    diagnostics: list[dict[str, str]]
    audit_summary: dict[str, Any]
    backtest_summary: dict[str, Any]
    technical_review: dict[str, Any]
    candidate_setups_defined: bool
    promotion_decision: str
    blocking_issues: list[str]
    run_log: list[dict[str, Any]]


@dataclass(slots=True)
class ReviewCheckpoint:
    name: CheckpointName
    instruction: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoopLimits:
    max_trials: int
    max_feature_changes: int
    max_threshold_changes: int


@dataclass(slots=True)
class WorkflowSnapshot:
    phase: str
    evidence_boundary: EvidenceBoundary
    approvals: dict[str, bool]
    experiment_counts: dict[str, int]
    blocking_issues: list[str]
