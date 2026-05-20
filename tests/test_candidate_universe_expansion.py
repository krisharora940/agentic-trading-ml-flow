from __future__ import annotations

import unittest

from trading_ml.candidate_universe_expansion import build_candidate_universe_expansion_space
from trading_ml.research_controller import build_candidate_universe_expansion_search_space


class CandidateUniverseExpansionTests(unittest.TestCase):
    def test_candidate_universe_space_requires_lineage_and_effective_sample_size(self) -> None:
        space = build_candidate_universe_expansion_space()

        self.assertEqual(space["family"], "candidate_universe_expansion")
        self.assertEqual(space["max_batch_trials"], len(space["variants"]))
        self.assertIn("candidate_lineage", space["required_governance"])
        self.assertIn("deduplication", space["required_governance"])
        self.assertIn("effective_sample_size_accounting", space["required_governance"])
        self.assertIn("holdout", space["disallowed"])

    def test_controller_exposes_candidate_universe_space(self) -> None:
        space = build_candidate_universe_expansion_search_space()

        names = {row["name"] for row in space["variants"]}
        self.assertIn("first_reclaim_only_baseline", names)
        self.assertIn("allow_multiple_same_direction_candidates", names)
        self.assertIn("extended_structure_zone", names)


if __name__ == "__main__":
    unittest.main()
