import unittest
from unittest import mock

from trading_ml.agent_nodes import audit_agent_node, diagnosis_agent_node, iteration_controller_node, search_controller_agent_node, setup_redesign_agent_node, translation_checkpoint_node
from trading_ml.agent_state import LoopLimits
from trading_ml.agent_workflow import build_agent_loop_state, pending_human_checkpoints, run_linear_stage3_pass
from trading_ml.langgraph_integration import route_after_data_steward_state, route_after_program_director_state


class AgentWorkflowTests(unittest.TestCase):
    def test_initial_agent_loop_state_has_expected_checkpoint_flags(self) -> None:
        state = build_agent_loop_state()
        self.assertIn("bnr_spec_approval", state["approvals"])
        self.assertEqual(state["phase"], "exploration")
        self.assertEqual(state["runtime_profile"], "standard")
        self.assertTrue(state["data_manifest_loaded"])
        self.assertEqual(state["stage2_config"]["symbol"], "MNQ")
        self.assertIn("program_state", state)
        self.assertIn("strategy_notes", state)
        self.assertIn("compute_budgets", state)
        self.assertIn("budget_usage", state)
        self.assertIn("research_backlog", state)
        self.assertIn("failure_memory", state)
        self.assertEqual(state["search_batch_status"], "pending")

    def test_initial_agent_loop_state_can_preapprove_bounded_cycle_checkpoints(self) -> None:
        state = build_agent_loop_state(preapproved_checkpoints=["label_approval", "search_space_approval"], runtime_profile="bounded_autonomous")
        self.assertFalse(state["approvals"]["bnr_spec_approval"])
        self.assertTrue(state["approvals"]["label_approval"])
        self.assertTrue(state["approvals"]["search_space_approval"])
        self.assertEqual(state["runtime_profile"], "bounded_autonomous")

    def test_initial_agent_loop_state_loads_persisted_cross_run_memory(self) -> None:
        with mock.patch(
            "trading_ml.agent_workflow.load_persisted_research_memory",
            return_value={
                "failure_memory": [{"memory_id": "FM-1"}],
                "research_action_history": [{"action_id": "feature"}],
                "desk_memory": [{"proposal_id": "DPROP-1"}],
            },
        ):
            state = build_agent_loop_state()
        self.assertEqual(state["failure_memory"][0]["memory_id"], "FM-1")
        self.assertEqual(state["research_action_history"][0]["action_id"], "feature")
        self.assertEqual(state["desk_memory"][0]["proposal_id"], "DPROP-1")

    def test_linear_stage3_pass_is_disabled(self) -> None:
        with self.assertRaises(RuntimeError):
            run_linear_stage3_pass()

    def test_pending_human_checkpoints_surface_payloads(self) -> None:
        state = build_agent_loop_state()
        state["checkpoints_pending"] = ["label_approval"]
        checkpoints = pending_human_checkpoints(state)
        names = {item["name"] for item in checkpoints}
        self.assertIn("label_approval", names)

    def test_search_controller_skips_family_already_executed_in_cycle(self) -> None:
        state = build_agent_loop_state()
        state["next_step_plan"] = {
            "selected_family": "subtype",
            "controller_override": {"active_family": "subtype"},
        }
        state["approvals"]["search_space_approval"] = True
        state["executed_research_family"] = "subtype"
        state["executed_family_cycle"] = state["research_cycle"]
        state["search_batch_status"] = "complete"
        state["search_results"] = {"family": "subtype", "trial_count": 2}
        limits = LoopLimits(max_trials=50, max_feature_changes=12, max_threshold_changes=10)
        with mock.patch("trading_ml.agent_nodes.run_governed_search") as run_search:
            result = search_controller_agent_node(state, limits)
        run_search.assert_not_called()
        self.assertEqual(result["execution_mode"], "diagnostic_only")
        self.assertEqual(result["search_results"]["family"], "subtype")

    def test_setup_redesign_agent_emits_planning_mandate(self) -> None:
        state = build_agent_loop_state()
        state["next_step_plan"] = {
            "benchmark_status": "exhausted_or_structurally_fragile",
            "why_selected": "same CPCV tail persisted",
            "rationale": {"persistent_tail_paths": ["cpcv_010"], "families_failed": ["model", "label", "sample_expansion"]},
        }
        diagnostic = {"status": "complete", "family": "market_state_setup_quality", "search_executed": False, "models_trained": 0}
        with mock.patch("trading_ml.agent_nodes.build_market_state_setup_quality_diagnostic", return_value=diagnostic):
            result = setup_redesign_agent_node(state)
        plan = result["setup_redesign_plan"]
        self.assertEqual(result["execution_mode"], "diagnostic_only")
        self.assertEqual(result["market_state_setup_quality_diagnostic"], diagnostic)
        self.assertEqual(plan["market_state_setup_quality_diagnostic"], diagnostic)
        self.assertEqual(plan["status"], "ready_for_setup_redesign")
        self.assertEqual(plan["bounded_search_budget"]["max_trials"], 4)
        self.assertEqual(plan["approval_checkpoint"], "setup_redesign_mandate_approval")
        self.assertEqual(plan["research_focus"]["new_abstraction"], "BNR geometry plus evolving intraday auction state and setup-quality interpretation")
        self.assertIn("market_state_quality_classifier", [row["name"] for row in plan["candidate_setup_hypotheses"]])
        self.assertIn("intraday_auction_state", [row["family"] for row in plan["latent_feature_families"]])
        family = plan["new_research_family"]
        self.assertEqual(family["family"], "market_state_setup_quality")
        self.assertEqual(family["priority_hypothesis"], "market_state_quality_classifier")
        self.assertEqual(family["holdout_status"], "locked")
        self.assertIn("candle_speed_volatility", family["candidate_features"])
        self.assertIn("setup_quality_label", family["candidate_labels"])
        self.assertIn("balanced_chop", [row["state"] for row in family["quality_states"]])
        self.assertIn("purged CPCV with changed worst-path signature", family["validation_plan"]["required_gates"])
        self.assertIn("reopening parked BNR benchmark geometry", family["bounded_search_budget"]["disallowed_knobs"])

    def test_iteration_controller_continues_to_new_hypothesis_after_freeze(self) -> None:
        state = build_agent_loop_state()
        state["research_backlog"] = [
            {"hypothesis_id": "H-00001", "family": "setup", "priority": 0.82},
            {"hypothesis_id": "H-00002", "family": "candidate_universe_expansion", "priority": 0.79},
        ]
        state["failure_memory"] = [
            {
                "family": "setup",
                "hypothesis_id": "H-00001",
                "failure_type": "cpcv_tail_path_fragility",
                "status": "freeze",
            }
        ]
        state["promotion_decision"] = "freeze"
        state["translation_summary"] = {"status": "pass"}
        state["audit_summary"] = {
            "cpcv": {"status": "fail"},
            "deflated_sharpe": {"status": "fail"},
            "walk_forward": {"status": "pass"},
            "purging": {"status": "pass"},
        }
        state["search_results"] = {"family": "setup", "trial_count": 6}
        result = iteration_controller_node(state)
        self.assertEqual(result["research_cycle"], 2)
        self.assertEqual(result["search_batch_status"], "pending")
        self.assertEqual(result["search_results"], {})
        self.assertEqual(result["executed_research_family"], "")
        self.assertEqual(result["translation_summary"], {})

    def test_search_controller_records_director_assigned_action_history(self) -> None:
        state = build_agent_loop_state()
        state["approvals"]["search_space_approval"] = True
        state["next_step_plan"] = {
            "selected_family": "setup",
            "assigned_research_action": "validation_failure_analysis",
            "hypothesis_id": "H-00001",
            "controller_override": {"active_family": "setup"},
        }
        state["stage2_config"]["source_path"] = "dummy"
        limits = LoopLimits(max_trials=50, max_feature_changes=12, max_threshold_changes=10)
        with mock.patch("trading_ml.agent_nodes.execute_research_action", return_value={"family": "research_diagnostics", "trial_count": 0, "batch_decision": "inform", "status": "complete"}) as run_action:
            with mock.patch("trading_ml.agent_nodes.append_action_history_entry") as append_action:
                result = search_controller_agent_node(state, limits)
        run_action.assert_called_once()
        append_action.assert_called_once()
        self.assertEqual(result["research_action_history"][-1]["action_id"], "validation_failure_analysis")

    def test_bounded_autonomous_profile_caps_governed_search_trials(self) -> None:
        state = build_agent_loop_state(runtime_profile="bounded_autonomous")
        state["approvals"]["search_space_approval"] = True
        state["next_step_plan"] = {
            "selected_family": "setup",
            "assigned_research_action": "setup",
            "hypothesis_id": "H-00001",
            "controller_override": {"active_family": "setup"},
        }
        state["stage2_config"]["source_path"] = "dummy"
        limits = LoopLimits(max_trials=50, max_feature_changes=12, max_threshold_changes=10)
        with mock.patch("trading_ml.agent_nodes.execute_research_action", return_value={"family": "setup", "trial_count": 2, "batch_decision": "revise", "status": "complete"}) as run_action:
            search_controller_agent_node(state, limits)
        controller = run_action.call_args.kwargs["controller_state"]
        self.assertEqual(controller["max_batch_trials"], 2)

    def test_desk_handoff_forces_tiny_first_governed_batch(self) -> None:
        state = build_agent_loop_state()
        state["approvals"]["search_space_approval"] = True
        state["next_step_plan"] = {
            "selected_family": "candidate_universe_expansion",
            "assigned_research_action": "candidate_universe_expansion",
            "hypothesis_id": "DPROP-1",
            "controller_override": {"active_family": "candidate_universe_expansion"},
            "search_budget": {"max_trials": 1},
            "desk_handoff": {"proposal_id": "DPROP-1", "first_governed_batch": True},
        }
        state["stage2_config"]["source_path"] = "dummy"
        limits = LoopLimits(max_trials=50, max_feature_changes=12, max_threshold_changes=10)
        with mock.patch("trading_ml.agent_nodes.execute_research_action", return_value={"family": "candidate_universe_expansion", "trial_count": 1, "batch_decision": "revise", "status": "complete"}) as run_action:
            search_controller_agent_node(state, limits)
        controller = run_action.call_args.kwargs["controller_state"]
        self.assertEqual(controller["max_batch_trials"], 1)
        self.assertEqual(controller["fast_variant_names"][:2], ["first_reclaim_only_baseline", "allow_delayed_reclaim"])
        self.assertEqual(controller["max_sessions"], 96)

    def test_desk_handoff_routes_to_governor_before_setup_redesign(self) -> None:
        state = build_agent_loop_state()
        state["next_step_plan"] = {
            "selected_family": "candidate_universe_expansion",
            "assigned_research_action": "candidate_universe_expansion",
            "benchmark_status": "exhausted_or_structurally_fragile",
            "approval_required": "search_space_approval",
            "controller_override": {"active_family": "candidate_universe_expansion"},
            "desk_handoff": {"proposal_id": "DPROP-1", "first_governed_batch": True},
        }
        self.assertEqual(route_after_program_director_state(state), "governor_agent")

    def test_desk_handoff_structural_family_skips_to_search_after_data_steward(self) -> None:
        state = build_agent_loop_state()
        state["next_step_plan"] = {
            "assigned_research_action": "candidate_universe_expansion",
            "desk_handoff": {"proposal_id": "DPROP-1", "first_governed_batch": True},
        }
        self.assertEqual(route_after_data_steward_state(state), "search_controller_agent")

    def test_diagnostic_action_skips_full_audit_budget_consumption(self) -> None:
        state = build_agent_loop_state()
        state["budget_usage"] = {"runtime_seconds": 0, "trials": 6, "full_validations": 2, "cpcv_runs": 2, "model_trains": 6}
        state["search_results"] = {
            "action": {"action_id": "cpcv_attribution", "callable_kind": "stateful_diagnostic_action"},
        }
        result = audit_agent_node(state)
        self.assertEqual(result["budget_usage"]["full_validations"], 2)
        self.assertEqual(result["budget_usage"]["cpcv_runs"], 2)
        self.assertEqual(result["audit_summary"]["research_diagnostics"]["action_id"], "cpcv_attribution")

    def test_structural_governed_action_skips_full_audit_budget_consumption(self) -> None:
        state = build_agent_loop_state()
        state["budget_usage"] = {"runtime_seconds": 0, "trials": 6, "full_validations": 2, "cpcv_runs": 2, "model_trains": 6}
        state["search_results"] = {
            "family": "candidate_universe_expansion",
            "governance": {"promotion_blocked": True, "models_trained": 0},
            "action": {"action_id": "candidate_universe_expansion", "callable_kind": "governed_research_cycle"},
        }
        result = audit_agent_node(state)
        self.assertEqual(result["budget_usage"]["full_validations"], 2)
        self.assertEqual(result["budget_usage"]["cpcv_runs"], 2)
        self.assertEqual(result["audit_summary"]["structural_research"]["family"], "candidate_universe_expansion")

    def test_diagnostic_action_skips_translation_analysis(self) -> None:
        state = build_agent_loop_state()
        state["search_results"] = {
            "action": {"action_id": "validation_failure_analysis", "callable_kind": "stateful_diagnostic_action"},
        }
        result = translation_checkpoint_node(state)
        self.assertEqual(result["translation_summary"]["status"], "inform")
        self.assertEqual(result["translation_summary"]["diagnostic_action"], "validation_failure_analysis")

    def test_structural_governed_action_skips_translation_analysis(self) -> None:
        state = build_agent_loop_state()
        state["search_results"] = {
            "family": "candidate_universe_expansion",
            "governance": {"promotion_blocked": True, "models_trained": 0},
            "action": {"action_id": "candidate_universe_expansion", "callable_kind": "governed_research_cycle"},
        }
        result = translation_checkpoint_node(state)
        self.assertEqual(result["translation_summary"]["status"], "inform")
        self.assertEqual(result["translation_summary"]["structural_action"], "candidate_universe_expansion")

    def test_diagnosis_agent_builds_attempts_and_failure_clusters(self) -> None:
        state = build_agent_loop_state()
        state["stage2_result"] = {
            "features_records": [
                {
                    "candidate_id": "c1",
                    "direction": "long",
                    "setup_subtype": "weak_reclaim",
                    "trigger_seconds_after_open": 180,
                    "break_efficiency_ratio": 0.2,
                    "reclaim_close_location": 0.3,
                    "reclaim_failure_count": 0,
                    "deepest_zone_retrace_fraction": 0.85,
                    "post_reclaim_close_strength": 0.1,
                },
                {
                    "candidate_id": "c2",
                    "direction": "long",
                    "setup_subtype": "weak_reclaim",
                    "trigger_seconds_after_open": 240,
                    "break_efficiency_ratio": 0.22,
                    "reclaim_close_location": 0.28,
                    "reclaim_failure_count": 0,
                    "deepest_zone_retrace_fraction": 0.82,
                    "post_reclaim_close_strength": 0.12,
                },
                {
                    "candidate_id": "c3",
                    "direction": "short",
                    "setup_subtype": "weak_reclaim",
                    "trigger_seconds_after_open": 260,
                    "break_efficiency_ratio": 0.25,
                    "reclaim_close_location": 0.26,
                    "reclaim_failure_count": 0,
                    "deepest_zone_retrace_fraction": 0.81,
                    "post_reclaim_close_strength": 0.11,
                },
            ],
            "labels_records": [
                {"candidate_id": "c1", "label": 0, "outcome": "stop", "pnl_r": -1.0, "bars_held": 3},
                {"candidate_id": "c2", "label": 0, "outcome": "stop", "pnl_r": -0.8, "bars_held": 2},
                {"candidate_id": "c3", "label": 0, "outcome": "stop", "pnl_r": -0.7, "bars_held": 2},
            ],
        }
        state["audit_summary"] = {
            "cpcv": {"status": "fail"},
            "walk_forward": {
                "status": "pass",
                "stitched_prediction_records": [
                    {"candidate_id": "c1", "session_date": "2026-01-02", "probability": 0.61, "prediction": 1},
                    {"candidate_id": "c2", "session_date": "2026-01-02", "probability": 0.59, "prediction": 1},
                    {"candidate_id": "c3", "session_date": "2026-01-03", "probability": 0.58, "prediction": 1},
                ],
            },
        }
        state["translation_summary"] = {"status": "pass"}
        with mock.patch("trading_ml.agent_nodes.append_failure_memory_entry") as append_failure:
            result = diagnosis_agent_node(state)
        append_failure.assert_called_once()
        self.assertEqual(len(result["bnr_attempts"]), 3)
        self.assertTrue(result["failure_clusters"])
        self.assertEqual(result["failure_clusters"][0]["recommended_family"], "setup")


if __name__ == "__main__":
    unittest.main()
