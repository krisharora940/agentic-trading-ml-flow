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
    run_id: str
    program_state: dict[str, Any]
    next_step_plan: dict[str, Any]
    research_director_summary: dict[str, Any]
    benchmark_status: str
    setup_redesign_plan: dict[str, Any]
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
    executed_research_family: str
    executed_family_cycle: int
    search_batch_status: str
    execution_mode: Literal["diagnostic_only", "cheap_search", "full_validation"]
    compute_budgets: dict[str, Any]
    budget_usage: dict[str, Any]
    route_decisions: list[dict[str, Any]]
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
    domain_priors: list[dict[str, Any]]
    research_backlog: list[dict[str, Any]]
    active_hypothesis: dict[str, Any]
    failure_memory: list[dict[str, Any]]
    research_action_history: list[dict[str, Any]]
    candidate_setups_defined: bool
    promotion_decision: str
    holdout_consumed: bool
    blocking_issue_records: list[dict[str, Any]]
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
    max_runtime_seconds: int = 1800
    max_full_validations: int = 3
    max_cpcv_runs: int = 3
    max_model_trains: int = 25
    reuse_artifacts: bool = True
    stop_on_budget_exhaustion: bool = True


@dataclass(slots=True)
class WorkflowSnapshot:
    phase: str
    evidence_boundary: EvidenceBoundary
    approvals: dict[str, bool]
    experiment_counts: dict[str, int]
    blocking_issues: list[str]
