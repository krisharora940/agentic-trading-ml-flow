from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from trading_ml.candidate_universe_expansion import (
    build_candidate_universe_expansion_space,
    run_candidate_universe_shortlist_diagnostic,
    _selected_variants_from_inventory,
    _resolve_variant_subset,
    _slice_bars_for_runtime,
)
from trading_ml.research_controller import (
    build_candidate_universe_expansion_search_space,
)


class CandidateUniverseExpansionTests(unittest.TestCase):
    def test_candidate_universe_space_requires_lineage_and_effective_sample_size(
        self,
    ) -> None:
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

    def test_resolve_variant_subset_includes_baseline(self) -> None:
        variants = _resolve_variant_subset(
            {"fast_variant_names": ["allow_delayed_reclaim"]}
        )
        names = [row["name"] for row in variants]
        self.assertEqual(names[0], "first_reclaim_only_baseline")
        self.assertIn("allow_delayed_reclaim", names)

    def test_slice_bars_for_runtime_limits_sessions(self) -> None:
        index = pd.to_datetime(
            [
                "2026-01-02 09:30:00-05:00",
                "2026-01-03 09:30:00-05:00",
                "2026-01-04 09:30:00-05:00",
            ]
        )
        bars = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=index)
        sliced = _slice_bars_for_runtime(bars, {"max_sessions": 2})
        self.assertEqual(len({ts.date() for ts in sliced.index}), 2)

    def test_selected_variants_from_inventory_extracts_shortlist(self) -> None:
        selected = _selected_variants_from_inventory(
            {
                "selected_for_next_stage": [
                    {"variant": "allow_delayed_reclaim"},
                    {"variant": "first_reclaim_only_baseline"},
                ]
            }
        )
        self.assertEqual(
            selected,
            ["allow_delayed_reclaim", "first_reclaim_only_baseline"],
        )

    def test_shortlist_diagnostic_marks_shortlist_only(self) -> None:
        with (
            patch(
                "trading_ml.candidate_universe_expansion._read_latest_candidate_universe_inventory",
                return_value={
                    "artifact_path": "reports/candidate_universe_expansion.json",
                    "selected_for_next_stage": [
                        {"variant": "allow_delayed_reclaim"},
                        {"variant": "first_reclaim_only_baseline"},
                    ],
                },
            ),
            patch(
                "trading_ml.candidate_universe_expansion.run_candidate_universe_labeling_diagnostic",
                return_value={"status": "complete", "inventory_gate": {}},
            ) as run_diag,
        ):
            result = run_candidate_universe_shortlist_diagnostic({})

        run_diag.assert_called_once()
        self.assertTrue(result["inventory_gate"]["shortlist_only"])


if __name__ == "__main__":
    unittest.main()
