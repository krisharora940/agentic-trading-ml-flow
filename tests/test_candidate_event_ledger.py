from __future__ import annotations

import unittest

from trading_ml.candidate_event_ledger import (
    build_candidate_event_ledger_key,
    candidate_from_ledger_record,
    candidate_to_ledger_record,
)
from trading_ml.stage2_bnr import BNRZone, CandidateSetup


class CandidateEventLedgerTests(unittest.TestCase):
    def test_candidate_roundtrip_preserves_identity(self) -> None:
        zone = BNRZone(
            symbol="MNQ",
            session_date="2026-01-02",
            zone_start="2026-01-02T09:30:00-05:00",
            zone_end="2026-01-02T09:30:30-05:00",
            decision_available_at="2026-01-02T09:31:00-05:00",
            high=101.0,
            low=99.0,
            midpoint=100.0,
            width=2.0,
            width_bps=200.0,
            source_timeframe="30s",
        )
        candidate = CandidateSetup(
            candidate_id="MNQ-1",
            symbol="MNQ",
            session_date="2026-01-02",
            zone=zone,
            break_time="2026-01-02T09:31:00-05:00",
            break_decision_time="2026-01-02T09:32:00-05:00",
            trigger_time="2026-01-02T09:33:00-05:00",
            decision_time="2026-01-02T09:33:30-05:00",
            direction="long",
            setup_type="break_reentry_reclaim_long",
            entry_reference_price=101.5,
            invalidation_reference_price=99.0,
            pivot_price=99.5,
            pivot_time="2026-01-02T09:32:30-05:00",
            flem_price=102.0,
            flem_time="2026-01-02T09:33:30-05:00",
            reentry_count=1,
            reclaim_count=1,
            feature_cutoff_time="2026-01-02T09:33:30-05:00",
            trace={"note": "test"},
        )

        record = candidate_to_ledger_record(candidate, variant_name="baseline")
        rebuilt = candidate_from_ledger_record(record)

        self.assertEqual(rebuilt.candidate_id, candidate.candidate_id)
        self.assertEqual(rebuilt.zone.high, candidate.zone.high)
        self.assertEqual(rebuilt.trace, candidate.trace)

    def test_ledger_key_is_stable_for_same_inputs(self) -> None:
        left = build_candidate_event_ledger_key(
            source_path="data/path.parquet",
            symbol="MNQ",
            timeframe="30s",
            variant_names=["b", "a"],
            session_dates=["2026-01-02", "2026-01-03"],
            generator_version="v1",
        )
        right = build_candidate_event_ledger_key(
            source_path="data/path.parquet",
            symbol="MNQ",
            timeframe="30s",
            variant_names=["a", "b"],
            session_dates=["2026-01-02", "2026-01-03"],
            generator_version="v1",
        )

        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
