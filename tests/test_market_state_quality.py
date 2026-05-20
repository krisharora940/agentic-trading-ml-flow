import unittest
from unittest import mock

import pandas as pd

from trading_ml.market_state_quality import build_market_state_setup_quality_diagnostic
from trading_ml.stage2_pipeline import Stage2Config


class MarketStateQualityTests(unittest.TestCase):
    def test_diagnostic_builds_labels_without_models_or_search(self) -> None:
        features = pd.DataFrame(
            [
                {
                    "candidate_id": "a",
                    "session_date": "2025-01-01",
                    "break_range_expansion": 2.0,
                    "break_efficiency_ratio": 0.9,
                    "break_body_fraction": 0.8,
                    "reg_trend_strength_30": 0.7,
                    "reg_chop_30": 0.1,
                    "pivot_symmetry": 0.8,
                    "post_reclaim_close_strength": 0.8,
                    "reclaim_count": 0,
                    "deepest_zone_retrace_fraction": 0.2,
                },
                {
                    "candidate_id": "b",
                    "session_date": "2025-01-02",
                    "break_range_expansion": 0.4,
                    "break_efficiency_ratio": 0.2,
                    "break_body_fraction": 0.2,
                    "reg_trend_strength_30": 0.1,
                    "reg_chop_30": 0.9,
                    "pivot_symmetry": 0.2,
                    "post_reclaim_close_strength": 0.1,
                    "reclaim_count": 3,
                    "deepest_zone_retrace_fraction": 0.8,
                },
            ]
        )
        labels = pd.DataFrame(
            [
                {"candidate_id": "a", "label": 1, "pnl_r": 1.5},
                {"candidate_id": "b", "label": 0, "pnl_r": -1.0},
            ]
        )
        state = {"stage2_config": {"source_path": "/tmp/exploration.parquet"}}
        config = Stage2Config(source_path="/tmp/exploration.parquet")
        with mock.patch(
            "trading_ml.market_state_quality._build_point_in_time_inputs",
            return_value=(features, labels, {"feature_audit": {"failed": 0, "issues": []}, "config": {}}),
        ):
            with mock.patch("trading_ml.market_state_quality._diagnostic_config", return_value=config):
                result = build_market_state_setup_quality_diagnostic(state)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["family"], "market_state_setup_quality")
        self.assertEqual(result["execution_mode"], "diagnostic_only")
        self.assertFalse(result["search_executed"])
        self.assertEqual(result["models_trained"], 0)
        self.assertEqual(result["holdout_status"], "locked")
        self.assertIn("candidate_counts_by_state", result)
        self.assertIn("pnl_by_state", result)
        self.assertEqual(result["cheap_state_policy_simulation"]["status"], "complete")
        self.assertEqual(len(result["cheap_state_policy_simulation"]["policy_variants"]), 7)
        self.assertEqual(result["leakage_audit"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
