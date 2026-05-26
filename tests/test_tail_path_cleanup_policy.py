import unittest

from trading_ml.tail_path_cleanup_policy import apply_tail_path_cleanup_policy


class TailPathCleanupPolicyTests(unittest.TestCase):
    def test_excludes_deep_retrace_records(self) -> None:
        records = [
            {
                "candidate_id": "1",
                "setup_subtype": "deep_retrace_repair",
                "probability": 0.8,
                "entry_time": "2026-01-01T14:32:00+00:00",
            },
            {
                "candidate_id": "2",
                "setup_subtype": "clean_break_continuation",
                "probability": 0.8,
                "entry_time": "2026-01-01T14:32:30+00:00",
            },
        ]
        filtered = apply_tail_path_cleanup_policy(
            records, policy_name="exclude_deep_retrace", threshold=0.45
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["candidate_id"], "2")

    def test_high_conf_cap_reduces_deep_retrace_probability_bucket(self) -> None:
        records = [
            {
                "candidate_id": "1",
                "setup_subtype": "deep_retrace_repair",
                "probability": 0.82,
                "entry_time": "2026-01-01T14:32:00+00:00",
            }
        ]
        filtered = apply_tail_path_cleanup_policy(
            records, policy_name="deep_retrace_high_conf_cap", threshold=0.45
        )
        self.assertEqual(filtered[0]["probability"], 0.64)

    def test_exact_time_bucket_policy_is_diagnostic_only(self) -> None:
        records = [
            {
                "candidate_id": "1",
                "setup_subtype": "clean_break_continuation",
                "probability": 0.82,
                "entry_time": "2026-01-01T14:32:00+00:00",
            }
        ]
        with self.assertRaises(ValueError):
            apply_tail_path_cleanup_policy(
                records, policy_name="exclude_1432_bucket", threshold=0.45
            )


if __name__ == "__main__":
    unittest.main()
