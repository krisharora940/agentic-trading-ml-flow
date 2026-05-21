import unittest
from unittest import mock

from trading_ml.bnr_research_desk import desk_director_node, event_librarian_node, failure_analyst_node, route_after_desk_director
from trading_ml.langgraph_integration import build_bnr_research_desk_initial_input, build_governor_state_from_desk_handoff, compile_bnr_research_desk_graph


def _synthetic_stage2_result() -> dict:
    return {
        "config": {"timezone": "America/New_York"},
        "data_quality": {"sessions": 3},
        "zone_count": 3,
        "candidate_count": 3,
        "features_records": [
            {
                "candidate_id": "c1",
                "session_date": "2026-01-02",
                "direction": "long",
                "setup_subtype": "clean_reclaim",
                "trigger_seconds_after_open": 120,
                "break_efficiency_ratio": 0.75,
                "reclaim_close_location": 0.82,
                "reclaim_failure_count": 1,
                "deepest_zone_retrace_fraction": 0.2,
                "post_reclaim_close_strength": 0.7,
            },
            {
                "candidate_id": "c2",
                "session_date": "2026-01-03",
                "direction": "long",
                "setup_subtype": "weak_reclaim",
                "trigger_seconds_after_open": 180,
                "break_efficiency_ratio": 0.2,
                "reclaim_close_location": 0.3,
                "reclaim_failure_count": 0,
                "deepest_zone_retrace_fraction": 0.1,
                "post_reclaim_close_strength": 0.1,
            },
            {
                "candidate_id": "c3",
                "session_date": "2026-01-04",
                "direction": "short",
                "setup_subtype": "weak_reclaim",
                "trigger_seconds_after_open": 240,
                "break_efficiency_ratio": 0.22,
                "reclaim_close_location": 0.28,
                "reclaim_failure_count": 0,
                "deepest_zone_retrace_fraction": 0.15,
                "post_reclaim_close_strength": 0.15,
            },
        ],
        "labels_records": [
            {"candidate_id": "c1", "label": 1, "outcome": "target", "pnl_r": 1.1, "bars_held": 4},
            {"candidate_id": "c2", "label": 0, "outcome": "timeout", "pnl_r": -0.2, "bars_held": 8},
            {"candidate_id": "c3", "label": 0, "outcome": "timeout", "pnl_r": -0.1, "bars_held": 9},
        ],
        "model_summary": {
            "prediction_records": [
                {"candidate_id": "c1", "probability": 0.73, "prediction": 1},
                {"candidate_id": "c2", "probability": 0.62, "prediction": 1},
                {"candidate_id": "c3", "probability": 0.58, "prediction": 1},
            ]
        },
    }


class BNRResearchDeskTests(unittest.TestCase):
    def test_event_librarian_and_failure_analyst_build_attempts_and_clusters(self) -> None:
        state = {
            "run_id": "bnr-desk-test",
            "research_cycle": 1,
            "phase": "exploration",
            "stage2_result": _synthetic_stage2_result(),
            "audit_summary": {},
            "desk_summary": {},
            "run_log": [],
        }
        event_update = event_librarian_node(state)
        self.assertEqual(len(event_update["bnr_attempts"]), 3)
        failure_update = failure_analyst_node({**state, **event_update})
        self.assertTrue(failure_update["failure_clusters"])
        self.assertEqual(failure_update["failure_clusters"][0]["family"], "no_follow_through")

    def test_desk_director_routes_no_follow_through_to_path_modeler(self) -> None:
        state = {
            "run_id": "bnr-desk-test",
            "research_cycle": 1,
            "phase": "exploration",
            "failure_clusters": [{"family": "no_follow_through", "recommended_family": "exit_behavior_research", "cluster_id": "fc-1"}],
            "desk_memory": [],
            "desk_summary": {},
            "run_log": [],
        }
        update = desk_director_node(state)
        self.assertEqual(update["desk_summary"]["desk_director"]["selected_node"], "path_modeler")
        self.assertEqual(route_after_desk_director({**state, **update}), "path_modeler")

    def test_desk_director_avoids_repeating_prior_proposal_family(self) -> None:
        state = {
            "run_id": "bnr-desk-test",
            "research_cycle": 1,
            "phase": "exploration",
            "failure_clusters": [{"family": "no_follow_through", "recommended_family": "exit_behavior_research", "cluster_id": "fc-1"}],
            "desk_memory": [{"proposal_family": "path_modeling", "proposal_id": "DPROP-old"}],
            "desk_summary": {},
            "run_log": [],
        }
        update = desk_director_node(state)
        self.assertEqual(update["desk_summary"]["desk_director"]["selected_node"], "exit_research_agent")

    def test_desk_director_avoids_repeating_family_already_executed_by_governor(self) -> None:
        state = {
            "run_id": "bnr-desk-test",
            "research_cycle": 1,
            "phase": "exploration",
            "failure_clusters": [{"family": "no_reclaim_edge", "recommended_family": "candidate_universe_expansion", "cluster_id": "fc-1"}],
            "desk_memory": [{"proposal_family": "eligibility"}, {"proposal_family": "setup"}],
            "research_action_history": [{"family": "candidate_universe_expansion", "action_id": "candidate_universe_expansion"}],
            "desk_summary": {},
            "run_log": [],
        }
        update = desk_director_node(state)
        self.assertEqual(update["desk_summary"]["desk_director"]["selected_node"], "feature_engineer")

    def test_bnr_research_desk_graph_runs_to_handoff(self) -> None:
        graph = compile_bnr_research_desk_graph()
        initial_state = build_bnr_research_desk_initial_input()
        initial_state["desk_memory"] = []
        with mock.patch("trading_ml.bnr_research_desk.data_steward_agent_node", return_value={"current_node": "desk_data_steward", "stage2_result": _synthetic_stage2_result(), "run_log": []}):
            result = graph.invoke(initial_state, config={"configurable": {"thread_id": "bnr-desk-graph-test"}})
        self.assertTrue(result["desk_proposals"])
        self.assertEqual(result["desk_proposals"][-1]["family"], "path_modeling")
        self.assertEqual(result["desk_summary"]["desk_memory_update"]["status"], "ready_for_governor_graph")

    def test_build_governor_state_from_desk_handoff_carries_desk_artifacts(self) -> None:
        desk_state = {
            "stage2_result": _synthetic_stage2_result(),
            "bnr_attempts": [{"attempt_id": "ATT-c1"}],
            "failure_clusters": [{"cluster_id": "FC-1"}],
            "desk_summary": {"desk_memory_update": {"status": "ready_for_governor_graph"}},
            "desk_proposals": [{"proposal_id": "DPROP-1", "family": "path_modeling"}],
            "desk_memory": [{"proposal": {"proposal_id": "DPROP-1"}}],
            "run_log": [{"actor": "desk_memory_update"}],
        }
        state = build_governor_state_from_desk_handoff(desk_state)
        self.assertEqual(state["desk_proposals"][-1]["proposal_id"], "DPROP-1")
        self.assertEqual(state["failure_clusters"][0]["cluster_id"], "FC-1")
        self.assertEqual(state["bnr_attempts"][0]["attempt_id"], "ATT-c1")
        self.assertEqual(state["runtime_profile"], "bounded_autonomous")
        self.assertTrue(state["approvals"]["search_space_approval"])

    def test_desk_memory_update_persists_cross_run_memory(self) -> None:
        state = {
            "run_id": "bnr-desk-test",
            "research_cycle": 1,
            "phase": "exploration",
            "failure_clusters": [{"family": "no_follow_through"}],
            "desk_proposals": [{"proposal_id": "DPROP-1", "family": "path_modeling"}],
            "desk_memory": [],
            "desk_summary": {},
            "run_log": [],
        }
        from trading_ml.bnr_research_desk import desk_memory_update_node

        with mock.patch("trading_ml.bnr_research_desk.append_desk_memory_entry") as append_memory:
            result = desk_memory_update_node(state)
        append_memory.assert_called_once()
        self.assertEqual(result["desk_memory"][-1]["proposal"]["proposal_id"], "DPROP-1")
        self.assertEqual(result["desk_memory"][-1]["proposal_family"], "path_modeling")


if __name__ == "__main__":
    unittest.main()
