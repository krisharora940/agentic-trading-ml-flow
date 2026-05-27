from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from trading_ml.manual_validity_prior import (
    augment_features_with_manual_validity_prior,
)
from trading_ml.stage2_bnr import BNRZone, CandidateSetup


class ManualValidityPriorTests(unittest.TestCase):
    def test_augments_features_with_manual_prior_columns(self) -> None:
        features = pd.DataFrame(
            [
                {
                    "candidate_id": "c1",
                    "session_date": "2026-01-02",
                    "direction_long": 1.0,
                    "zone_width": 2.0,
                    "pre_trigger_range": 3.0,
                },
                {
                    "candidate_id": "c2",
                    "session_date": "2026-01-03",
                    "direction_long": 0.0,
                    "zone_width": 2.5,
                    "pre_trigger_range": 2.0,
                },
            ]
        )
        candidates = [
            _candidate("c1", "2026-01-02T09:32:30-05:00", "long", "2026-01-02"),
            _candidate("c2", "2026-01-03T09:33:00-05:00", "short", "2026-01-03"),
        ]
        manual = pd.DataFrame(
            [
                {
                    "open_date": "2026-01-02",
                    "session_date": "2026-01-02",
                    "direction": "long",
                    "entry_anchor_timestamp": pd.Timestamp("2026-01-02T09:32:30-05:00"),
                    "setup_valid_normalized": "yes",
                    "quality_bucket": "high_quality",
                    "manual_validity_label": 1,
                    "manual_high_quality_label": 1,
                },
                {
                    "open_date": "2026-01-03",
                    "session_date": "2026-01-03",
                    "direction": "short",
                    "entry_anchor_timestamp": pd.Timestamp("2026-01-03T09:33:00-05:00"),
                    "setup_valid_normalized": "no",
                    "quality_bucket": "avoid",
                    "manual_validity_label": 0,
                    "manual_high_quality_label": 0,
                },
            ]
        )

        with patch(
            "trading_ml.manual_validity_prior._load_manual_examples",
            return_value=manual,
        ):
            augmented, summary = augment_features_with_manual_validity_prior(
                features, candidates
            )

        self.assertIn("manual_validity_probability", augmented.columns)
        self.assertIn("manual_high_quality_probability", augmented.columns)
        self.assertIn("manual_example_matched", augmented.columns)
        self.assertEqual(summary.status, "available")
        self.assertEqual(summary.matched_candidate_count, 2)


def _candidate(
    candidate_id: str, decision_time: str, direction: str, session_date: str
) -> CandidateSetup:
    zone = BNRZone(
        symbol="MNQ",
        session_date=session_date,
        zone_start=f"{session_date}T09:30:00-05:00",
        zone_end=f"{session_date}T09:30:30-05:00",
        decision_available_at=f"{session_date}T09:31:00-05:00",
        high=101.0,
        low=99.0,
        midpoint=100.0,
        width=2.0,
        width_bps=200.0,
        source_timeframe="30s",
    )
    return CandidateSetup(
        candidate_id=candidate_id,
        symbol="MNQ",
        session_date=session_date,
        zone=zone,
        break_time=decision_time,
        break_decision_time=decision_time,
        trigger_time=decision_time,
        decision_time=decision_time,
        direction=direction,  # type: ignore[arg-type]
        setup_type=f"break_reentry_reclaim_{direction}",
        entry_reference_price=100.0,
        invalidation_reference_price=99.0,
        pivot_price=99.5,
        pivot_time=decision_time,
        flem_price=101.0,
        flem_time=decision_time,
        reentry_count=1,
        reclaim_count=1,
        feature_cutoff_time=decision_time,
        trace={},
    )


if __name__ == "__main__":
    unittest.main()
