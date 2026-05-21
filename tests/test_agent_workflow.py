import unittest
from unittest import mock

from trading_ml.agent_nodes import audit_agent_node, iteration_controller_node, search_controller_agent_node, setup_redesign_agent_node, translation_checkpoint_node
from trading_ml.agent_state import LoopLimits
from trading_ml.agent_workflow import build_agent_loop_state, pending_human_checkpoints, run_linear_stage3_pass


class AgentWorkflowTests(unittest.TestCase):
    def test_initial_agent_loop_state_has_expected_checkpoint_flags(self) -> None:
        state = build_agent_loop_state()
        self.assertIn("bnr_spec_approval", state["approvals"])
        self.assertEqual(state["phase"], "exploration")
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
        state = build_agent_loop_state(preapproved_checkpoints=["label_approval", "search_space_approval"])
        self.assertFalse(state["approvals"]["bnr_spec_approval"])
        self.assertTrue(state["approvals"]["label_approval"])
        self.assertTrue(state["approvals"]["search_space_approval"])

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
            result = search_controller_agent_node(state, limits)
        run_action.assert_called_once()
        self.assertEqual(result["research_action_history"][-1]["action_id"], "validation_failure_analysis")

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

    def test_diagnostic_action_skips_translation_analysis(self) -> None:
        state = build_agent_loop_state()
        state["search_results"] = {
            "action": {"action_id": "validation_failure_analysis", "callable_kind": "stateful_diagnostic_action"},
        }
        result = translation_checkpoint_node(state)
        self.assertEqual(result["translation_summary"]["status"], "inform")
        self.assertEqual(result["translation_summary"]["diagnostic_action"], "validation_failure_analysis")


if __name__ == "__main__":
    unittest.main()
