from __future__ import annotations

import unittest

from trading_ml.exit_behavior_research import build_exit_behavior_research_space
from trading_ml.research_controller import build_exit_behavior_research_search_space


class ExitBehaviorResearchTests(unittest.TestCase):
    def test_exit_behavior_research_space_is_governed_and_bounded(self) -> None:
        space = build_exit_behavior_research_space()

        self.assertEqual(space["family"], "exit_behavior_research")
        self.assertIn("trade_path_diagnostics", space["stages"])
        self.assertIn("bounded_replay_existing_entries", space["stages"])
        self.assertIn("model_training", space["disallowed_knobs"])
        self.assertIn("holdout", space["disallowed_knobs"])
        self.assertEqual(space["max_batch_trials"], len(space["bounded_replay_variants"]))

    def test_research_controller_exposes_exit_behavior_space(self) -> None:
        space = build_exit_behavior_research_search_space()

        self.assertEqual(space["family"], "exit_behavior_research")
        self.assertIn("liquidity_rejection_exit", space["candidate_exit_families"])


if __name__ == "__main__":
    unittest.main()
