import unittest
from unittest import mock

from trading_ml.agent_nodes import search_controller_agent_node, setup_redesign_agent_node
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
        self.assertEqual(state["search_batch_status"], "pending")

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


if __name__ == "__main__":
    unittest.main()
