import unittest

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.search import run_governed_search
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.validation_audit import build_validation_audit


class ValidationAuditTests(unittest.TestCase):
    def test_validation_audit_surfaces_walk_forward_purging_and_multiple_testing(self) -> None:
        state = build_agent_loop_state()
        stage2_result = run_stage2_research_engine(Stage2Config(**state["stage2_config"]))
        search_results = run_governed_search(state["stage2_config"])
        audit = build_validation_audit(stage2_result, search_results)
        self.assertIn(audit["walk_forward"]["status"], {"pass", "fail", "pending"})
        self.assertIn(audit["walk_forward"].get("backend", "custom"), {"custom", "ml4t_diagnostic"})
        self.assertIn(audit["purging"]["status"], {"pass", "fail", "pending"})
        self.assertIn(audit["multiple_testing"]["status"], {"pass", "fail", "pending"})
        self.assertIn(audit["overfitting"], {"pass", "fail", "pending"})


if __name__ == "__main__":
    unittest.main()
