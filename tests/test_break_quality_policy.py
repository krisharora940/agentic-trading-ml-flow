import unittest

from trading_ml.break_quality_policy import apply_break_quality_policy


class BreakQualityPolicyTests(unittest.TestCase):
    def test_balanced_deep_efficiency_gate_only_filters_targeted_subtypes(self) -> None:
        stitched = [
            {"candidate_id": "a", "setup_subtype": "balanced_reclaim_continuation", "probability": 0.60},
            {"candidate_id": "b", "setup_subtype": "deep_retrace_repair", "probability": 0.60},
            {"candidate_id": "c", "setup_subtype": "clean_break_continuation", "probability": 0.60},
        ]
        features = [
            {"candidate_id": "a", "break_efficiency_ratio": 0.25},
            {"candidate_id": "b", "break_efficiency_ratio": 0.10},
            {"candidate_id": "c", "break_efficiency_ratio": 0.05},
        ]

        filtered = apply_break_quality_policy(
            stitched,
            features,
            policy_name="balanced_deep_eff_ge_0.20",
            threshold=0.45,
        )

        ids = {row["candidate_id"] for row in filtered}
        self.assertIn("a", ids)
        self.assertNotIn("b", ids)
        self.assertIn("c", ids)


if __name__ == "__main__":
    unittest.main()
