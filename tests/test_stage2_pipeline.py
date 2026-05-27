import unittest

import pandas as pd

from trading_ml.stage2_bnr import calculate_bnr_zones, generate_breakout_candidates
from trading_ml.stage2_features import build_feature_matrix
from trading_ml.stage2_labeling import label_candidates


class Stage2PipelineTests(unittest.TestCase):
    def test_bnr_zone_candidate_label_and_feature_audit_are_point_in_time(self) -> None:
        index = pd.date_range(
            "2026-01-02 09:30:00", periods=12, freq="30s", tz="America/New_York"
        )
        bars = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104, 103, 102, 104, 105, 106, 108, 109],
                "high": [102, 103, 104, 105, 106, 104, 105, 106, 107, 109, 111, 112],
                "low": [99, 100, 101, 102, 102, 101, 101, 103, 104, 105, 107, 108],
                "close": [101, 102, 103, 104, 103, 102, 104, 105, 106, 108, 110, 111],
                "volume": [
                    1000,
                    900,
                    1200,
                    1100,
                    1000,
                    950,
                    925,
                    900,
                    880,
                    860,
                    840,
                    820,
                ],
            },
            index=index,
        )

        zones = calculate_bnr_zones(bars)
        self.assertEqual(len(zones), 1)
        self.assertEqual(zones[0].decision_available_at, "2026-01-02T09:31:00-05:00")

        candidates = generate_breakout_candidates(
            bars, zones, earliest_trigger_time="09:32:00"
        )
        self.assertGreaterEqual(len(candidates), 1)
        self.assertGreaterEqual(
            pd.Timestamp(candidates[0].decision_time),
            pd.Timestamp(zones[0].decision_available_at),
        )

        labels = label_candidates(bars, candidates, horizon_bars=4, target_multiple=1.0)
        self.assertEqual(len(labels), len(candidates))

        features, audits = build_feature_matrix(bars, candidates)
        self.assertTrue(all(audit.status == "pass" for audit in audits))
        self.assertIn("first_break_wick_only", features.columns)
        self.assertIn("continuation_displacement_ratio", features.columns)
        self.assertIn("break_close_distance_to_zone", features.columns)
        self.assertIn("reclaim_close_location", features.columns)
        self.assertIn("retrace_at_entry", features.columns)
        self.assertIn("zone_to_extrema_atr", features.columns)
        self.assertTrue(
            {"eng_rsi", "eng_atr"} <= set(features.columns)
            or {"reg_vol_10", "reg_trend_10"} <= set(features.columns)
        )

    def test_candidate_requires_opposite_pullback_bar_before_reclaim(self) -> None:
        index = pd.date_range(
            "2026-01-02 09:30:00", periods=8, freq="30s", tz="America/New_York"
        )
        bars = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0, 104.2, 104.4, 104.6],
                "high": [102.0, 103.0, 104.0, 105.0, 105.2, 105.0, 105.4, 105.6],
                "low": [99.0, 100.0, 101.0, 102.0, 102.8, 103.4, 103.8, 104.2],
                "close": [101.0, 102.0, 103.0, 104.0, 104.2, 104.4, 104.6, 105.0],
                "volume": [1000, 950, 925, 900, 875, 850, 825, 800],
            },
            index=index,
        )

        zones = calculate_bnr_zones(bars)
        candidates = generate_breakout_candidates(
            bars, zones, earliest_trigger_time="09:32:00"
        )

        self.assertEqual(candidates, [])

    def test_event_driven_engine_can_emit_multiple_attempts_per_direction(self) -> None:
        index = pd.date_range(
            "2026-01-02 09:30:00", periods=18, freq="30s", tz="America/New_York"
        )
        bars = pd.DataFrame(
            {
                "open": [
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.6,
                    102.9,
                    104.4,
                    104.8,
                    102.8,
                    104.5,
                    104.9,
                    105.0,
                    105.2,
                    105.4,
                    105.6,
                    105.8,
                    106.0,
                    106.2,
                ],
                "high": [
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    104.8,
                    104.6,
                    105.0,
                    105.0,
                    104.7,
                    105.2,
                    105.4,
                    105.6,
                    105.8,
                    106.0,
                    106.2,
                    106.4,
                    106.6,
                    106.8,
                ],
                "low": [
                    99.0,
                    100.0,
                    101.0,
                    102.0,
                    102.7,
                    102.8,
                    104.1,
                    102.7,
                    102.7,
                    104.2,
                    104.4,
                    104.6,
                    104.8,
                    105.0,
                    105.2,
                    105.4,
                    105.6,
                    105.8,
                ],
                "close": [
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    102.9,
                    104.4,
                    104.8,
                    102.8,
                    104.5,
                    104.9,
                    105.1,
                    105.3,
                    105.5,
                    105.7,
                    105.9,
                    106.1,
                    106.3,
                    106.5,
                ],
                "volume": [1000] * 18,
            },
            index=index,
        )

        zones = calculate_bnr_zones(bars)
        candidates = generate_breakout_candidates(
            bars,
            zones,
            earliest_trigger_time="09:32:00",
            latest_trigger_time="10:30:00",
            max_candidates_per_direction=3,
            candidate_engine="event_driven_v1",
        )

        self.assertGreaterEqual(len(candidates), 2)
        self.assertTrue(
            all(
                candidate.trace.get("candidate_engine") == "event_driven_v1"
                for candidate in candidates
            )
        )


if __name__ == "__main__":
    unittest.main()
