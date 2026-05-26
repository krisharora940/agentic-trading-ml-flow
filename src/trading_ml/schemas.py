from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class EvidenceWindow:
    start: str
    end: str


@dataclass(slots=True)
class EvidenceBoundary:
    mode: str
    exploration: EvidenceWindow
    validation: EvidenceWindow
    holdout: EvidenceWindow
    notes: str = ""


@dataclass(slots=True)
class DataManifestEntry:
    symbol: str
    timeframe: str
    date_start: str
    date_end: str
    source_path: str
    timezone: str
    schema: str
    quality_flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BNRSpec:
    name: str
    timezone: str
    zone_start: str
    zone_end: str
    decision_start: str
    candidate_description: str = ""


@dataclass(slots=True)
class LabelSpec:
    target_name: str
    positive_class: str
    negative_class: str
    horizon_bars: int
    stop_rule: str
    target_rule: str
    timeout_rule: str


@dataclass(slots=True)
class FeatureSpec:
    name: str
    families: list[str] = field(default_factory=list)
    decision_time_only: bool = True
    notes: str = ""


@dataclass(slots=True)
class ModelSpec:
    family: str
    objective: str
    calibration: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestSummary:
    engine: str
    costs_applied: bool
    slippage_applied: bool
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class AuditSummary:
    leakage_check: str = "not-run"
    point_in_time_check: str = "not-run"
    validation_check: str = "not-run"
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BlockingIssue:
    code: str
    severity: str
    category: str
    node: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BNRAttempt:
    attempt_id: str
    candidate_id: str
    session_date: str
    direction: str
    setup_subtype: str
    setup_state: str
    environment_state: str
    time_bucket: str
    probability_bucket: str
    executed: bool
    label: int | None = None
    prediction: int | None = None
    probability: float | None = None
    pnl_r: float | None = None
    outcome: str = "unknown"
    failure_reason: str = "unknown"
    path_class: str = "unknown"
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FailureCluster:
    cluster_id: str
    family: str
    rows: int
    avg_pnl_r: float
    avg_probability: float
    dominant_subtype: str
    dominant_setup_state: str
    dominant_environment_state: str
    dominant_time_bucket: str
    recommended_family: str
    recommended_focus: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StateTransition:
    from_state: str
    to_state: str
    trigger: str
    sample_size: int = 0
    transition_probability: float | None = None
    persistence_bars: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContinuationProfile:
    state: str
    sample_size: int
    continuation_rate: float
    failure_rate: float
    avg_pnl_r: float | None = None
    median_persistence_bars: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FailureProfile:
    failure_family: str
    state: str
    sample_size: int
    avg_pnl_r: float | None = None
    dominant_path_class: str = "unknown"
    dominant_repair_state: str = "unknown"
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StateOntology:
    ontology_id: str
    version: int
    primary_modeling_target: str
    bnr_role: str
    state_definitions: dict[str, dict[str, Any]] = field(default_factory=dict)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    continuation_profiles: list[dict[str, Any]] = field(default_factory=list)
    failure_profiles: list[dict[str, Any]] = field(default_factory=list)
    persistence_statistics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DeskProposal:
    proposal_id: str
    node: str
    family: str
    claim: str
    hypothesis: str = ""
    action_id: str = ""
    target_failure_cluster: str | None = None
    target_market_state: str | None = None
    target_setup_state: str | None = None
    target_environment_state: str | None = None
    target_path_class: str | None = None
    proposed_features: list[str] = field(default_factory=list)
    parameter_knobs: list[str] = field(default_factory=list)
    expected_metric_delta: dict[str, Any] = field(default_factory=dict)
    allowable_knobs: list[str] = field(default_factory=list)
    forbidden_knobs: list[str] = field(default_factory=list)
    support_requirements: list[str] = field(default_factory=list)
    falsification_rule: str = ""
    expected_target_metrics: list[str] = field(default_factory=list)
    kill_criteria: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResearchActionPlan:
    plan_id: str
    proposal_id: str
    action_id: str
    family: str
    objective: str
    target_failure_cluster: str | None = None
    expected_metric_delta: dict[str, Any] = field(default_factory=dict)
    allowable_knobs: list[str] = field(default_factory=list)
    forbidden_knobs: list[str] = field(default_factory=list)
    support_requirements: list[str] = field(default_factory=list)
    falsification_rule: str = ""
    kill_criteria: list[str] = field(default_factory=list)
    controller_state: dict[str, Any] = field(default_factory=dict)
    base_config_overrides: dict[str, Any] = field(default_factory=dict)
    validation_scope: str = "governor_only"
    requires_governor_validation: bool = True
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResearchActionResult:
    result_id: str
    action_id: str
    proposal_id: str
    status: str
    family: str = ""
    trial_count: int = 0
    batch_decision: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    raw_summary: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MarginalEvidence:
    proposal_id: str
    action_id: str
    status: str
    decision: str = "inform"
    net_delta_vs_baseline: float | None = None
    robustness_delta: float | None = None
    cpcv_delta: float | None = None
    dsr_delta: float | None = None
    calibration_delta: float | None = None
    worst_path_loss_delta: float | None = None
    sample_delta: int | None = None
    notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FamilyExhaustionRecord:
    family: str
    status: str
    reason: str = ""
    consecutive_cycles: int = 0
    accepted_without_robustness: int = 0
    last_proposal_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MarginalImprovementTracker:
    family: str
    proposal_id: str
    action_id: str
    metric_deltas: dict[str, Any] = field(default_factory=dict)
    has_robustness_improvement: bool = False
    decision: str = "inform"
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResearchBranchStatus:
    family: str
    status: str
    exhaustion: dict[str, Any] = field(default_factory=dict)
    marginal_improvement: dict[str, Any] = field(default_factory=dict)
    next_allowed_action: str = ""
    notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RedTeamReview:
    proposal_id: str
    status: str
    blocked_reasons: list[str] = field(default_factory=list)
    critiques: list[str] = field(default_factory=list)
    required_revisions: list[str] = field(default_factory=list)
    boundary_check: str = "not-run"
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperimentRecord:
    experiment_id: str
    hypothesis: str
    config_ref: str
    data_slice: dict[str, str]
    result: dict[str, Any]
    decision: str
    phase: str
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class ProjectState:
    phase: str
    evidence_boundary: EvidenceBoundary
    data_manifests: list[DataManifestEntry] = field(default_factory=list)
    bnr_spec: BNRSpec | None = None
    labels: list[LabelSpec] = field(default_factory=list)
    features: list[FeatureSpec] = field(default_factory=list)
    models: list[ModelSpec] = field(default_factory=list)
    backtests: list[BacktestSummary] = field(default_factory=list)
    audits: list[AuditSummary] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
