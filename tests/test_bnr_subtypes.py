import unittest

from trading_ml.bnr_subtypes import (
    classify_candidate_subtype,
    filter_candidates_by_subtype,
)
from trading_ml.stage2_bnr import BNRZone, CandidateSetup


class BNRSubtypesTests(unittest.TestCase):
    def test_classify_candidate_subtype(self) -> None:
        zone = BNRZone(
            symbol="MNQ",
            session_date="2026-01-02",
            zone_start="2026-01-02T09:30:00-05:00",
            zone_end="2026-01-02T09:30:30-05:00",
            decision_available_at="2026-01-02T09:31:00-05:00",
            high=100.0,
            low=99.0,
            midpoint=99.5,
            width=1.0,
            width_bps=100.0,
            source_timeframe="30s",
        )
        candidate = CandidateSetup(
            candidate_id="c1",
            symbol="MNQ",
            session_date="2026-01-02",
            zone=zone,
            break_time="2026-01-02T09:31:00-05:00",
            break_decision_time="2026-01-02T09:32:00-05:00",
            trigger_time="2026-01-02T09:32:00-05:00",
            decision_time="2026-01-02T09:32:30-05:00",
            direction="long",
            setup_type="x",
            entry_reference_price=100.5,
            invalidation_reference_price=99.0,
            pivot_price=99.9,
            pivot_time="2026-01-02T09:32:00-05:00",
            flem_price=101.0,
            flem_time="2026-01-02T09:31:30-05:00",
            reentry_count=1,
            reclaim_count=1,
            feature_cutoff_time="2026-01-02T09:32:30-05:00",
            trace={
                "first_break_close_confirmed": 1.0,
                "first_break_wick_only": 0.0,
                "deepest_zone_retrace_fraction": 0.2,
                "post_reclaim_close_strength": 0.7,
                "continuation_displacement_ratio": 0.8,
            },
        )
        self.assertEqual(
            classify_candidate_subtype(candidate), "clean_break_continuation"
        )
        self.assertEqual(
            len(filter_candidates_by_subtype([candidate], "clean_break_continuation")),
            1,
        )


if __name__ == "__main__":
    unittest.main()
