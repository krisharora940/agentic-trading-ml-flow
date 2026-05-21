from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "run_bnr_desk_governor_workflow.py"
    spec = importlib.util.spec_from_file_location("run_bnr_desk_governor_workflow", module_path)
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
        "desk_proposals": [{"proposal_id": "DPROP-1", "family": "path_modeling"}],
    }
    governor_result = {
        "run_id": "bnr-governor",
        "next_step_plan": {
            "selected_family": "exit_behavior_research",
            "assigned_research_action": "exit_behavior_research",
            "hypothesis_id": "DPROP-1",
        },
        "promotion_decision": "freeze",
        "research_action_history": [{"action_id": "exit_behavior_research", "proposal_id": "DPROP-1"}],
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
    assert summary["suppressed_runtime_stderr"]["line_count"] == 2
