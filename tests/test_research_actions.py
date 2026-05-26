import unittest
from unittest import mock

from trading_ml.research_actions import (
    available_research_actions,
    execute_research_action,
)
from trading_ml.research_action_registry import (
    build_research_action_plan,
    execute_research_action_plan,
)


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
        self.assertEqual(
            registry["candidate_universe_expansion"].callable_kind,
            "governed_research_cycle",
        )

    def test_execute_research_action_wraps_governed_cycle_with_action_metadata(
        self,
    ) -> None:
        with mock.patch(
            "trading_ml.research_actions.run_governed_research_cycle",
            return_value={"family": "candidate_universe_expansion", "trial_count": 1},
        ) as run_cycle:
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

    def test_v2_action_registry_wraps_existing_research_actions(self) -> None:
        proposal = {
            "proposal_id": "DPROP-1",
            "family": "path_modeling",
            "claim": "Measure continuation decay by auction state.",
        }
        plan = build_research_action_plan(proposal, {"stage2_config_overrides": {}})
        self.assertEqual(plan["action_id"], "continuation_policy_search")
        self.assertTrue(plan["requires_governor_validation"])
        self.assertEqual(
            plan["expected_metric_delta"]["primary"],
            "auction_state_continuation_validity",
        )
        self.assertTrue(plan["allowable_knobs"])
        self.assertTrue(plan["allowed_policy_atoms"])
        self.assertTrue(plan["search_mechanics"])
        self.assertEqual(
            plan["doctrine"]["primary_modeling_target"],
            "auction_state_continuation_validity",
        )
        self.assertIn("holdout_access", plan["forbidden_knobs"])
        self.assertTrue(plan["support_requirements"])
        self.assertTrue(plan["falsification_rule"])
        self.assertTrue(plan["kill_criteria"])
        with mock.patch(
            "trading_ml.research_action_registry.execute_research_action",
            return_value={
                "family": "exit_behavior_research",
                "status": "complete",
                "trial_count": 1,
                "batch_decision": "inform",
            },
        ):
            result = execute_research_action_plan(
                plan,
                {
                    "stage2_config": {
                        "source_path": "x",
                        "symbol": "MNQ",
                        "timeframe": "30s",
                    }
                },
            )
        self.assertEqual(result["action_id"], "continuation_policy_search")
        self.assertEqual(result["status"], "complete")


if __name__ == "__main__":
    unittest.main()
