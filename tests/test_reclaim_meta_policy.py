import unittest

from trading_ml.reclaim_meta_policy import apply_reclaim_meta_policy


class ReclaimMetaPolicyTests(unittest.TestCase):
    def test_reclaim_meta_policy_filters_only_scoped_subtypes(self) -> None:
        records = [
            {
                "candidate_id": "a",
                "setup_subtype": "balanced_reclaim_continuation",
                "reclaim_close_location": 0.10,
                "post_reclaim_close_strength": 0.40,
                "reclaim_body_strength": 0.50,
            },
            {
                "candidate_id": "b",
                "setup_subtype": "clean_break_continuation",
                "reclaim_close_location": 0.20,
                "post_reclaim_close_strength": 0.40,
                "reclaim_body_strength": 0.50,
            },
            {
                "candidate_id": "c",
                "setup_subtype": "weak_follow_through",
                "reclaim_close_location": 0.00,
                "post_reclaim_close_strength": 0.00,
                "reclaim_body_strength": 0.00,
            },
        ]
        filtered = apply_reclaim_meta_policy(
            records, policy_name="balanced_clean_reclaim_close_ge_0.15"
        )
        ids = {row["candidate_id"] for row in filtered}
        self.assertNotIn("a", ids)
        self.assertIn("b", ids)
        self.assertIn("c", ids)


if __name__ == "__main__":
    unittest.main()
