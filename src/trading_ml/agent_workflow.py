from __future__ import annotations

from dataclasses import asdict
from typing import Any

from trading_ml.agent_config import load_agent_loop_config
from trading_ml.config import load_bnr_config, load_databento_manifest
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
from trading_ml.agent_state import AgentLoopState, LoopLimits
from trading_ml.bootstrap import build_initial_project_state
from trading_ml.research_controller import load_controller_config


NODE_SEQUENCE = [
    "governor_agent",
    "cto_agent",
    "data_steward_agent",
    "bnr_research_agent",
    "labeling_agent",
    "feature_agent",
    "model_agent",
    "backtest_agent",
    "search_controller_agent",
    "audit_agent",
    "translation_checkpoint",
    "diagnosis_agent",
    "promotion_decision",
    "iteration_controller",
]


def _select_stage2_source_path(manifest: dict[str, Any]) -> str | None:
    files = manifest.get("files", [])
    thirty_second = [entry for entry in files if entry.get("timeframe") == "30s"]
    if not thirty_second:
        return None
    chosen = max(thirty_second, key=lambda entry: (entry.get("sessions", 0), entry.get("rows", 0)))
    return chosen.get("source_path")


def build_agent_loop_state() -> AgentLoopState:
    project_state = build_initial_project_state()
    config = load_agent_loop_config()
    manifest = load_databento_manifest()
    bnr_config = load_bnr_config()
    stage2_source_path = _select_stage2_source_path(manifest)
    blocking_issues = list(project_state.blocking_issues)
    if manifest.get("symbol"):
        blocking_issues = [issue for issue in blocking_issues if issue != "No symbols configured yet."]
    if manifest.get("files"):
        blocking_issues = [issue for issue in blocking_issues if issue != "No Databento manifests loaded yet."]
    if bnr_config.get("setup", {}).get("name") == "BNR":
        blocking_issues = [issue for issue in blocking_issues if issue != "No candidate setup rules defined yet."]
    phase = "exploration" if not blocking_issues else config["graph"]["default_phase"]
    return AgentLoopState(
        phase=phase,
        current_node="start",
        evidence_boundary=project_state.evidence_boundary.to_dict() if hasattr(project_state.evidence_boundary, "to_dict") else asdict(project_state.evidence_boundary),
        bnr_spec=bnr_config,
        label_spec={},
        feature_spec={},
        feature_diagnostics={},
        model_spec={},
        stage2_config=(
            {
                "source_path": stage2_source_path,
                "symbol": manifest.get("symbol", "MNQ"),
                "timeframe": "30s",
                "timezone": manifest.get("timezone", "America/New_York"),
                "earliest_trigger_time": bnr_config["phases"]["entry"]["earliest_entry_time"],
                "latest_trigger_time": "11:00:00",
                "horizon_bars": bnr_config["label_v1"]["horizon_bars"],
                "stop_multiple": bnr_config["label_v1"]["stop_r"],
                "target_multiple": bnr_config["label_v1"]["target_r"],
                "break_buffer_points": bnr_config["phases"]["break"]["minimum_break_magnitude_points"],
                "spec_name": bnr_config["setup"]["name"],
            }
            if stage2_source_path
            else {}
        ),
        stage2_result={},
        controller_state=load_controller_config(),
        search_space={},
        search_results={},
        translation_summary={},
        frozen_benchmark={},
        approvals={name: False for name, enabled in config["checkpoints"].items() if enabled},
        checkpoints_pending=[],
        experiment_counts={"trials": 0, "feature_changes": 0, "threshold_changes": 0},
        research_cycle=1,
        max_research_cycles=3,
        diagnostics=[],
        audit_summary={},
        backtest_summary={},
        technical_review={},
        candidate_setups_defined=False,
        promotion_decision="revise",
        data_manifest_loaded=bool(manifest.get("files")),
        data_manifest=manifest,
        blocking_issues=blocking_issues,
        run_log=[],
    )


def build_loop_limits() -> LoopLimits:
    config = load_agent_loop_config()
    return LoopLimits(**config["limits"])


def run_linear_stage3_pass(state: AgentLoopState | None = None) -> AgentLoopState:
    current = dict(state or build_agent_loop_state())
    limits = build_loop_limits()
    for node_name in NODE_SEQUENCE:
        if node_name == "governor_agent":
            current.update(governor_agent_node(current))
        elif node_name == "cto_agent":
            current.update(cto_agent_node(current))
        elif node_name == "data_steward_agent":
            current.update(data_steward_agent_node(current))
        elif node_name == "bnr_research_agent":
            current.update(bnr_research_agent_node(current))
        elif node_name == "labeling_agent":
            current.update(labeling_agent_node(current))
        elif node_name == "feature_agent":
            current.update(feature_agent_node(current))
        elif node_name == "model_agent":
            current.update(model_agent_node(current))
        elif node_name == "backtest_agent":
            current.update(backtest_agent_node(current))
        elif node_name == "search_controller_agent":
            current.update(search_controller_agent_node(current, limits))
        elif node_name == "audit_agent":
            current.update(audit_agent_node(current))
        elif node_name == "translation_checkpoint":
            current.update(translation_checkpoint_node(current))
        elif node_name == "diagnosis_agent":
            current.update(diagnosis_agent_node(current))
        elif node_name == "promotion_decision":
            current.update(promotion_decision_node(current))
        elif node_name == "iteration_controller":
            current.update(iteration_controller_node(current))
    return AgentLoopState(**current)


def pending_human_checkpoints(state: AgentLoopState) -> list[dict[str, Any]]:
    pending = []
    for name in state.get("checkpoints_pending", []):
        if not state.get("approvals", {}).get(name, False):
            pending.append(asdict(checkpoint_payload(name, state)))
    return pending
