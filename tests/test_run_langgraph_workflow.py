from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "tools" / "run_langgraph_workflow.py"
    )
    spec = importlib.util.spec_from_file_location("run_langgraph_workflow", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_run_summary_includes_core_research_decision(tmp_path):
    module = _load_module()
    stderr_path = tmp_path / "runtime.stderr"
    stderr_path.write_text("warning one\nwarning two\n")
    result = {
        "run_id": "bnr-test",
        "research_cycle": 1,
        "research_director_summary": {"assigned_research_action": "setup"},
        "next_step_plan": {"selected_family": "setup"},
        "active_hypothesis": {"hypothesis_id": "H-1", "claim": "Test claim"},
        "search_batch_status": "complete",
        "search_results": {
            "trial_count": 6,
            "batch_decision": "revise",
            "family": "setup",
        },
        "translation_summary": {"translation_status": "pass"},
        "promotion_decision": "freeze",
        "blocking_issues": [],
        "run_log": [
            {
                "actor": "promotion_decision",
                "payload": {
                    "decision": "freeze",
                    "promotion_gate": {"cpcv_status": "fail"},
                },
            },
            {"actor": "iteration_controller", "payload": {"continue_iteration": False}},
        ],
    }

    summary = module._build_run_summary(result, stderr_path)

    assert summary["run_id"] == "bnr-test"
    assert summary["selected_family"] == "setup"
    assert summary["assigned_research_action"] == "setup"
    assert summary["active_hypothesis_id"] == "H-1"
    assert summary["promotion_decision"] == "freeze"
    assert summary["promotion_gate"]["cpcv_status"] == "fail"
    assert summary["suppressed_runtime_stderr"]["line_count"] == 2
