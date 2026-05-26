import unittest

from trading_ml.bnr_attempts import build_bnr_attempts
from trading_ml.failure_clusters import build_failure_clusters


class BNRAttemptTests(unittest.TestCase):
    def test_build_bnr_attempts_merges_stage2_and_walk_forward_artifacts(self) -> None:
        stage2_result = {
            "features_records": [
                {
                    "candidate_id": "c1",
                    "direction": "long",
                    "setup_subtype": "clean_reclaim",
                    "trigger_seconds_after_open": 120,
                    "break_efficiency_ratio": 0.7,
                    "reclaim_close_location": 0.8,
                    "reclaim_failure_count": 1,
                    "deepest_zone_retrace_fraction": 0.2,
                    "post_reclaim_close_strength": 0.7,
                    "reg_high_vol_state": 1,
                    "reg_trending_state": 1,
                },
                {
                    "candidate_id": "c2",
                    "direction": "long",
                    "setup_subtype": "weak_reclaim",
                    "trigger_seconds_after_open": 180,
                    "break_efficiency_ratio": 0.2,
                    "reclaim_close_location": 0.3,
                    "reclaim_failure_count": 0,
                    "deepest_zone_retrace_fraction": 0.85,
                    "post_reclaim_close_strength": 0.1,
                    "reg_high_vol_state": 0,
                    "reg_trending_state": 0,
                },
                {
                    "candidate_id": "c3",
                    "direction": "short",
                    "setup_subtype": "weak_reclaim",
                    "trigger_seconds_after_open": 240,
                    "break_efficiency_ratio": 0.25,
                    "reclaim_close_location": 0.25,
                    "reclaim_failure_count": 0,
                    "deepest_zone_retrace_fraction": 0.2,
                    "post_reclaim_close_strength": 0.12,
                    "reg_high_vol_state": 1,
                    "reg_trending_state": 0,
                },
                {
                    "candidate_id": "c4",
                    "direction": "short",
                    "setup_subtype": "weak_reclaim",
                    "trigger_seconds_after_open": 260,
                    "break_efficiency_ratio": 0.22,
                    "reclaim_close_location": 0.28,
                    "reclaim_failure_count": 0,
                    "deepest_zone_retrace_fraction": 0.18,
                    "post_reclaim_close_strength": 0.12,
                    "reg_high_vol_state": 1,
                    "reg_trending_state": 0,
                },
            ],
            "labels_records": [
                {
                    "candidate_id": "c1",
                    "label": 1,
                    "outcome": "target",
                    "pnl_r": 1.0,
                    "bars_held": 4,
                },
                {
                    "candidate_id": "c2",
                    "label": 0,
                    "outcome": "stop",
                    "pnl_r": -1.0,
                    "bars_held": 3,
                },
                {
                    "candidate_id": "c3",
                    "label": 0,
                    "outcome": "stop",
                    "pnl_r": -0.8,
                    "bars_held": 2,
                },
                {
                    "candidate_id": "c4",
                    "label": 0,
                    "outcome": "stop",
                    "pnl_r": -0.9,
                    "bars_held": 2,
                },
            ],
        }
        stitched = [
            {
                "candidate_id": "c1",
                "session_date": "2026-01-02",
                "probability": 0.71,
                "prediction": 1,
            },
            {
                "candidate_id": "c2",
                "session_date": "2026-01-02",
                "probability": 0.62,
                "prediction": 1,
            },
            {
                "candidate_id": "c3",
                "session_date": "2026-01-03",
                "probability": 0.58,
                "prediction": 1,
            },
            {
                "candidate_id": "c4",
                "session_date": "2026-01-03",
                "probability": 0.57,
                "prediction": 1,
            },
        ]

        attempts = build_bnr_attempts(stage2_result, stitched)
        self.assertEqual(len(attempts), 4)
        self.assertEqual(attempts[0]["time_bucket"], "early_open")
        self.assertEqual(attempts[0]["path_class"], "runner")
        self.assertEqual(attempts[0]["setup_state"], "continuation")
        self.assertEqual(attempts[0]["environment_state"], "trend_expansion")
        self.assertEqual(attempts[1]["failure_reason"], "deep_retrace_failure")
        self.assertEqual(attempts[1]["setup_state"], "repair")
        self.assertEqual(attempts[2]["environment_state"], "volatile_chop")
        self.assertEqual(attempts[2]["setup_state"], "weak_confirmation")

        clusters = build_failure_clusters(attempts)
        self.assertTrue(clusters)
        families = {cluster["family"] for cluster in clusters}
        self.assertIn("weak_continuation", families)
        volatile_cluster = next(
            cluster for cluster in clusters if cluster["family"] == "weak_continuation"
        )
        self.assertEqual(
            volatile_cluster["dominant_environment_state"], "volatile_chop"
        )
        self.assertEqual(volatile_cluster["dominant_setup_state"], "weak_confirmation")


if __name__ == "__main__":
    unittest.main()
