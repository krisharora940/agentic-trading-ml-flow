import unittest
from unittest import mock

from trading_ml.research_actions import available_research_actions, execute_research_action


class ResearchActionTests(unittest.TestCase):
    def test_registry_exposes_callable_one_off_research_capabilities(self) -> None:
        registry = available_research_actions()
        self.assertIn("candidate_universe_expansion", registry)
        self.assertIn("exit_behavior_research", registry)
        self.assertIn("tail_path_cleanup", registry)
        self.assertIn("setup_redesign", registry)
        self.assertIn("validation_failure_analysis", registry)
        self.assertIn("cpcv_attribution", registry)
        self.assertIn("domain_prior_ingestion", registry)
        self.assertEqual(registry["candidate_universe_expansion"].callable_kind, "governed_research_cycle")

    def test_execute_research_action_wraps_governed_cycle_with_action_metadata(self) -> None:
        with mock.patch("trading_ml.research_actions.run_governed_research_cycle", return_value={"family": "candidate_universe_expansion", "trial_count": 1}) as run_cycle:
            result = execute_research_action(
                "candidate_universe_expansion",
                base_config={"source_path": "x", "symbol": "MNQ", "timeframe": "30s"},
                controller_state={"active_family": "candidate_universe_expansion"},
            )
        run_cycle.assert_called_once()
        self.assertEqual(result["action"]["action_id"], "candidate_universe_expansion")
        self.assertEqual(result["family"], "candidate_universe_expansion")

    def test_execute_domain_prior_ingestion_returns_backlog(self) -> None:
        result = execute_research_action(
            "domain_prior_ingestion",
            base_config={"source_path": "x", "symbol": "MNQ", "timeframe": "30s"},
            controller_state={"active_family": "setup"},
            state={"failure_memory": [], "stage2_result": {}},
        )
        self.assertEqual(result["action"]["action_id"], "domain_prior_ingestion")
        self.assertTrue(result["domain_priors"])
        self.assertTrue(result["research_backlog"])


if __name__ == "__main__":
    unittest.main()
