import unittest

from trading_ml.subtype_policy import apply_subtype_policy


class SubtypePolicyTests(unittest.TestCase):
    def test_apply_subtype_policy_supports_exclusions_and_threshold_overrides(self) -> None:
        records = [
            {"candidate_id": "a", "setup_subtype": "weak_follow_through", "probability": 0.5},
            {"candidate_id": "b", "setup_subtype": "clean_break_continuation", "probability": 0.5},
            {"candidate_id": "c", "setup_subtype": "deep_retrace_repair", "probability": 0.56},
        ]
        filtered = apply_subtype_policy(
            records,
            default_threshold=0.45,
            threshold_overrides={"weak_follow_through": 0.6, "deep_retrace_repair": 0.55},
        )
        ids = {row["candidate_id"] for row in filtered}
        self.assertIn("b", ids)
        self.assertIn("c", ids)
        self.assertNotIn("a", ids)


if __name__ == "__main__":
    unittest.main()
