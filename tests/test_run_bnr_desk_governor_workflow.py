from __future__ import annotations

import importlib.util
from pathlib import Path
import time


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run_bnr_desk_governor_workflow.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_bnr_desk_governor_workflow", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_combined_summary_links_desk_and_governor(tmp_path):
    module = _load_module()
    stderr_path = tmp_path / "runtime.stderr"
    stderr_path.write_text("warning one\nwarning two\n")
    desk_result = {
        "run_id": "bnr-desk",
        "desk_summary": {
            "desk_director": {"selected_node": "path_modeler"},
            "desk_memory_update": {"status": "ready_for_governor_graph"},
        },
        "desk_proposals": [
            {
                "proposal_id": "DPROP-1",
                "family": "path_modeling",
                "action_id": "continuation_policy_search",
                "claim": "Continuation validity depends on follow-through shape.",
                "allowable_knobs": ["continuation_gate"],
            }
        ],
        "research_action_plan": {
            "plan_id": "PLAN-1",
            "proposal_id": "DPROP-1",
            "action_id": "continuation_policy_search",
            "objective": "Tighten continuation gating.",
            "search_mechanics": ["ablation_pack", "robust_window_rescore"],
            "controller_state": {"active_family": "exit_behavior_research"},
        },
    }
    governor_result = {
        "run_id": "bnr-governor",
        "next_step_plan": {
            "selected_family": "exit_behavior_research",
            "assigned_research_action": "exit_behavior_research",
            "hypothesis_id": "DPROP-1",
        },
        "promotion_decision": "freeze",
        "research_action_history": [
            {
                "action_id": "exit_behavior_research",
                "proposal_id": "DPROP-1",
                "batch_decision": "revise",
                "best_trial": {
                    "trial_id": "trial-1",
                    "overrides": {"continuation_policy": "combined_conservative"},
                    "positive_path_rate": 0.6,
                },
            }
        ],
        "blocking_issues": [],
        "run_log": [
            {"actor": "promotion_decision", "payload": {"decision": "freeze"}},
        ],
    }

    summary = module._build_combined_summary(desk_result, governor_result, stderr_path)

    assert summary["desk_selected_node"] == "path_modeler"
    assert summary["desk_proposal_id"] == "DPROP-1"
    assert summary["governor_selected_family"] == "exit_behavior_research"
    assert summary["governor_assigned_action"] == "exit_behavior_research"
    assert (
        summary["run_review"]["proposal"]["action_id"] == "continuation_policy_search"
    )
    assert summary["run_review"]["plan"]["plan_id"] == "PLAN-1"
    assert (
        summary["run_review"]["latest_action"]["best_trial"]["overrides"][
            "continuation_policy"
        ]
        == "combined_conservative"
    )
    assert summary["suppressed_runtime_stderr"]["line_count"] == 2


def test_run_governor_stops_early_on_completed_search_controller():
    module = _load_module()

    class FakeGraph:
        def stream(self, state, config=None, stream_mode=None):
            yield {"current_node": "governor_agent"}
            yield {
                "current_node": "search_controller_agent",
                "search_batch_status": "complete",
                "next_step_plan": {"selected_family": "setup"},
            }
            yield {"current_node": "audit_agent"}

    result = module._run_governor(
        FakeGraph(), {"current_node": "start"}, {}, summary_only=True
    )
    assert result["current_node"] == "search_controller_agent"
    assert result["search_batch_status"] == "complete"


def test_run_desk_stops_early_on_memory_update():
    module = _load_module()

    class FakeGraph:
        def stream(self, state, config=None, stream_mode=None):
            yield {"current_node": "desk_data_steward"}
            yield {
                "current_node": "desk_memory_update",
                "desk_summary": {
                    "desk_memory_update": {"status": "ready_for_governor_graph"}
                },
            }
            yield {"current_node": "done"}

    result = module._run_desk(
        FakeGraph(), {"current_node": "start"}, {}, summary_only=True
    )
    assert result["current_node"] == "desk_memory_update"


def test_run_governor_falls_back_to_invoke_for_full_mode():
    module = _load_module()

    class FakeGraph:
        def invoke(self, state, config=None):
            return {"current_node": "done", "promotion_decision": "freeze"}

    result = module._run_governor(
        FakeGraph(), {"current_node": "start"}, {}, summary_only=False
    )
    assert result["current_node"] == "done"


def test_run_desk_falls_back_to_invoke_for_full_mode():
    module = _load_module()

    class FakeGraph:
        def invoke(self, state, config=None):
            return {"current_node": "desk_memory_update"}

    result = module._run_desk(
        FakeGraph(), {"current_node": "start"}, {}, summary_only=False
    )
    assert result["current_node"] == "desk_memory_update"


def test_run_governor_returns_latest_state_on_summary_timeout():
    module = _load_module()

    class FakeGraph:
        def stream(self, state, config=None, stream_mode=None):
            yield {"current_node": "governor_agent"}
            time.sleep(0.2)
            yield {"current_node": "data_steward_agent"}

    result = module._run_governor(
        FakeGraph(),
        {"current_node": "start"},
        {},
        summary_only=True,
        summary_timeout_seconds=0.05,
    )
    assert result["current_node"] == "governor_agent"
