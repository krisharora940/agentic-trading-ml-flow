import unittest

from trading_ml.agent_nodes import promotion_decision_node
from trading_ml.agent_workflow import run_linear_stage3_pass


class GovernanceGateTests(unittest.TestCase):
    def test_linear_runner_is_disabled(self) -> None:
        with self.assertRaises(RuntimeError):
            run_linear_stage3_pass()

    def test_promotion_freezes_when_translation_pending(self) -> None:
        state = {
            "approvals": {"frozen_spec_approval": True},
            "audit_summary": {
                "leakage": "pass",
                "robustness": "pending",
                "walk_forward": {"status": "pass"},
                "cpcv": {"status": "pass"},
                "purging": {"status": "pass"},
                "overfitting": "pass",
                "multiple_testing": {"status": "pass", "promotable_method": False},
                "model_diagnostics": {"calibration_review": {"status": "pass"}},
            },
            "search_results": {"best_trial": {"net_avg_pnl_r": 0.1}},
            "translation_summary": {"status": "pending"},
            "blocking_issues": [],
        }
        result = promotion_decision_node(state)
        self.assertEqual(result["promotion_decision"], "freeze")

    def test_promotion_rejects_when_purging_fails(self) -> None:
        state = {
            "approvals": {"frozen_spec_approval": True},
            "audit_summary": {
                "leakage": "pass",
                "robustness": "pending",
                "walk_forward": {"status": "pass"},
                "cpcv": {"status": "pass"},
                "purging": {"status": "fail"},
                "overfitting": "pass",
                "multiple_testing": {"status": "pass", "promotable_method": True},
                "model_diagnostics": {"calibration_review": {"status": "pass"}},
            },
            "search_results": {"best_trial": {"net_avg_pnl_r": 0.1}},
            "translation_summary": {"status": "pass"},
            "blocking_issues": [],
        }
        result = promotion_decision_node(state)
        self.assertEqual(result["promotion_decision"], "reject")


if __name__ == "__main__":
    unittest.main()
