import unittest

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.search import run_governed_search
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.validation_audit import _multiple_testing_check, build_validation_audit


class ValidationAuditTests(unittest.TestCase):
    def test_multiple_testing_uses_canonical_familywise_method(self) -> None:
        search_results = {
            "trial_count": 4,
            "ranked_trials": [
                {"trial_id": "trial-001", "net_delta_vs_baseline": 0.01},
                {"trial_id": "trial-002", "net_delta_vs_baseline": 0.015},
                {"trial_id": "trial-003", "net_delta_vs_baseline": 0.02},
                {"trial_id": "trial-004", "net_delta_vs_baseline": 0.08},
            ],
            "accepted_trial": {
                "trial_id": "trial-004",
                "net_delta_vs_baseline": 0.08,
                "roc_auc_delta_vs_baseline": 0.01,
            },
        }
        audit = _multiple_testing_check(search_results)
        self.assertEqual(audit["method"], "familywise_bootstrap_max_delta")
        self.assertTrue(audit["promotable_method"])
        self.assertLessEqual(audit["familywise_pvalue"], 0.05)

    def test_validation_audit_surfaces_walk_forward_purging_and_multiple_testing(self) -> None:
        state = build_agent_loop_state()
        stage2_result = run_stage2_research_engine(Stage2Config(**state["stage2_config"]))
        search_results = run_governed_search(state["stage2_config"])
        audit = build_validation_audit(stage2_result, search_results)
        self.assertIn(audit["walk_forward"]["status"], {"pass", "fail", "pending"})
        self.assertIn(audit["walk_forward"].get("backend", "custom"), {"custom", "ml4t_diagnostic"})
        self.assertIn(audit["cpcv"]["status"], {"pass", "fail", "pending"})
        self.assertIn(audit["purging"]["status"], {"pass", "fail", "pending"})
        self.assertIn(audit["multiple_testing"]["status"], {"pass", "fail", "pending"})
        self.assertIn(audit["random_signal_plumbing"]["status"], {"pass", "fail", "pending"})
        self.assertIn(audit["overfitting"], {"pass", "fail", "pending"})


if __name__ == "__main__":
    unittest.main()
