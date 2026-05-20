from __future__ import annotations

from dataclasses import asdict
from typing import Any
from uuid import uuid4

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
    program_director_node,
    promotion_decision_node,
    search_controller_agent_node,
    strategy_intake_agent_node,
    translation_checkpoint_node,
)
from trading_ml.agent_state import AgentLoopState, LoopLimits
from trading_ml.bootstrap import build_initial_project_state
from trading_ml.evidence_sources import select_manifest_source_path
from trading_ml.research_program import build_program_state
from trading_ml.research_controller import load_controller_config


NODE_SEQUENCE = [
    "strategy_intake_agent",
    "program_director",
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
    "program_director",
    "diagnosis_agent",
    "promotion_decision",
    "iteration_controller",
]
def build_agent_loop_state() -> AgentLoopState:
    project_state = build_initial_project_state()
    config = load_agent_loop_config()
    manifest = load_databento_manifest()
    bnr_config = load_bnr_config()
    stage2_source_path = select_manifest_source_path(manifest, timeframe="30s", boundary_role="exploration")
    if stage2_source_path is None:
        stage2_source_path = select_manifest_source_path(manifest, timeframe="30s")
    blocking_issues = list(project_state.blocking_issues)
    if manifest.get("symbol"):
        blocking_issues = [issue for issue in blocking_issues if issue != "No symbols configured yet."]
    if manifest.get("files"):
        blocking_issues = [issue for issue in blocking_issues if issue != "No Databento manifests loaded yet."]
    if bnr_config.get("setup", {}).get("name") == "BNR":
        blocking_issues = [issue for issue in blocking_issues if issue != "No candidate setup rules defined yet."]
    phase = "exploration" if not blocking_issues else config["graph"]["default_phase"]
    return AgentLoopState(
        run_id=f"bnr-{uuid4().hex[:12]}",
        program_state=build_program_state(),
        next_step_plan={},
        strategy_notes=(
            "BNR trades the 9:30-9:30:59 opening zone on MNQ. "
            "The break matters less than the reclaim and continuation quality. "
            "We care about price action around the pivot, reclaim strength, time since open, and opening context."
        ),
        research_intake={},
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
                "earliest_trigger_time": bnr_config.get("frozen_benchmark", {}).get("setup_earliest_trigger_time", bnr_config["phases"]["entry"]["earliest_entry_time"]),
                "latest_trigger_time": "11:00:00",
                "horizon_bars": bnr_config.get("frozen_benchmark", {}).get("setup_horizon_bars", bnr_config["label_v1"]["horizon_bars"]),
                "stop_multiple": bnr_config["label_v1"]["stop_r"],
                "target_multiple": bnr_config.get("frozen_benchmark", {}).get("setup_target_multiple", bnr_config["label_v1"]["target_r"]),
                "break_buffer_points": bnr_config.get("frozen_benchmark", {}).get("setup_break_buffer_points", bnr_config["phases"]["break"]["minimum_break_magnitude_points"]),
                "spec_name": bnr_config["setup"]["name"],
                "model_family": bnr_config.get("frozen_benchmark", {}).get("model_family", "linear_baseline"),
                "feature_family": bnr_config.get("frozen_benchmark", {}).get("feature_family", "all_features"),
            }
            if stage2_source_path
            else {}
        ),
        stage2_result={},
        controller_state=load_controller_config(),
        search_space={},
        search_results={},
        executed_research_family="",
        executed_family_cycle=0,
        search_batch_status="pending",
        execution_mode="full_validation",
        compute_budgets={
            "max_runtime_seconds": int(config["limits"].get("max_runtime_seconds", 1800)),
            "max_trials": int(config["limits"].get("max_trials", 50)),
            "max_full_validations": int(config["limits"].get("max_full_validations", 3)),
            "max_cpcv_runs": int(config["limits"].get("max_cpcv_runs", 3)),
            "max_model_trains": int(config["limits"].get("max_model_trains", 25)),
            "reuse_artifacts": bool(config["limits"].get("reuse_artifacts", True)),
            "stop_on_budget_exhaustion": bool(config["limits"].get("stop_on_budget_exhaustion", True)),
        },
        budget_usage={
            "runtime_seconds": 0,
            "trials": 0,
            "full_validations": 0,
            "cpcv_runs": 0,
            "model_trains": 0,
        },
        route_decisions=[],
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
        holdout_consumed=False,
        data_manifest_loaded=bool(manifest.get("files")),
        data_manifest=manifest,
        blocking_issue_records=[],
        blocking_issues=blocking_issues,
        run_log=[],
    )


def build_loop_limits() -> LoopLimits:
    config = load_agent_loop_config()
    return LoopLimits(**config["limits"])


def run_linear_stage3_pass(state: AgentLoopState | None = None) -> AgentLoopState:
    raise RuntimeError("Linear runner disabled. Use compile_bnr_langgraph().")


def pending_human_checkpoints(state: AgentLoopState) -> list[dict[str, Any]]:
    pending = []
    for name in state.get("checkpoints_pending", []):
        if not state.get("approvals", {}).get(name, False):
            pending.append(asdict(checkpoint_payload(name, state)))
    return pending
