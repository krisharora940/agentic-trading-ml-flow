import unittest

from trading_ml.failure_analysis import build_failure_map


class FailureAnalysisTests(unittest.TestCase):
    def test_build_failure_map_groups_by_month_subtype_and_regime(self) -> None:
        stage2_result = {
            "features_records": [
                {
                    "candidate_id": "a",
                    "setup_subtype": "balanced_reclaim_continuation",
                    "reg_high_vol_state": 1.0,
                    "reg_trending_state": 0.0,
                    "trigger_seconds_after_open": 120.0,
                    "break_body_fraction": 0.6,
                    "break_efficiency_ratio": 0.5,
                    "reclaim_close_location": 0.8,
                    "reclaim_failure_count": 0.0,
                }
            ],
            "labels_records": [
                {"candidate_id": "a", "label": 0, "outcome": "stop", "pnl_r": -1.0, "bars_held": 3}
            ],
        }
        stitched = [
            {"candidate_id": "a", "session_date": "2026-01-05", "probability": 0.6, "label": 0, "pnl_r": -1.0}
        ]
        execution = {"equity_curve": [{"candidate_id": "a"}]}
        result = build_failure_map(stage2_result, stitched, execution)
        self.assertEqual(result["status"], "complete")
        self.assertGreaterEqual(len(result["by_month"]), 1)
        self.assertGreaterEqual(len(result["by_subtype"]), 1)
        executed_failure = result["executed_failures_by_subtype"][0]
        self.assertAlmostEqual(executed_failure["avg_break_efficiency_ratio"], 0.5)
        self.assertAlmostEqual(executed_failure["avg_reclaim_close_location"], 0.8)


if __name__ == "__main__":
    unittest.main()
