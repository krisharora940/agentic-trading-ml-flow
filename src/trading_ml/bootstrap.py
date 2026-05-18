from __future__ import annotations

from trading_ml.config import load_evidence_boundary_config, load_global_config
from trading_ml.schemas import BNRSpec, EvidenceBoundary, EvidenceWindow, ProjectState


def build_initial_project_state() -> ProjectState:
    global_config = load_global_config()
    boundary_config = load_evidence_boundary_config()

    boundary = EvidenceBoundary(
        mode=boundary_config["boundary"]["mode"],
        notes=boundary_config["boundary"]["notes"],
        exploration=EvidenceWindow(**boundary_config["exploration"]),
        validation=EvidenceWindow(**boundary_config["validation"]),
        holdout=EvidenceWindow(**boundary_config["holdout"]),
    )

    bnr_spec = BNRSpec(
        name=global_config["project"]["primary_setup"],
        timezone=global_config["project"]["timezone"],
        zone_start="09:30:00",
        zone_end="09:30:59",
        decision_start="09:31:00",
        candidate_description="Candidate setup definition pending Stage 2 research engine.",
    )

    return ProjectState(
        phase="foundation",
        evidence_boundary=boundary,
        bnr_spec=bnr_spec,
        blocking_issues=[
            "No symbols configured yet.",
            "No Databento manifests loaded yet.",
            "No candidate setup rules defined yet.",
        ],
    )
