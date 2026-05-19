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
