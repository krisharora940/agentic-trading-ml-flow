import unittest

from trading_ml.agent_workflow import build_agent_loop_state, pending_human_checkpoints, run_linear_stage3_pass


class AgentWorkflowTests(unittest.TestCase):
    def test_initial_agent_loop_state_has_expected_checkpoint_flags(self) -> None:
        state = build_agent_loop_state()
        self.assertIn("bnr_spec_approval", state["approvals"])
        self.assertEqual(state["phase"], "exploration")
        self.assertTrue(state["data_manifest_loaded"])
        self.assertEqual(state["stage2_config"]["symbol"], "MNQ")

    def test_linear_stage3_pass_produces_promotion_decision_and_logs(self) -> None:
        state = run_linear_stage3_pass()
        self.assertIn(state["promotion_decision"], {"reject", "revise", "freeze", "advance_to_validation"})
        self.assertGreater(len(state["run_log"]), 0)
        self.assertIn(state["current_node"], {"promotion_decision", "iteration_controller"})
        self.assertIn("stage2_result", state)

    def test_pending_human_checkpoints_surface_payloads(self) -> None:
        state = run_linear_stage3_pass()
        checkpoints = pending_human_checkpoints(state)
        names = {item["name"] for item in checkpoints}
        self.assertIn("label_approval", names)


if __name__ == "__main__":
    unittest.main()
