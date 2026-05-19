import unittest

from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.translation_policy import allow_signal_for_regime, compute_position_size


class TranslationPolicyTests(unittest.TestCase):
    def test_confidence_linear_size_increases_with_probability(self) -> None:
        small = compute_position_size(0.46, threshold=0.45, policy_name="confidence_linear_v1")
        large = compute_position_size(0.80, threshold=0.45, policy_name="confidence_linear_v1")
        self.assertGreaterEqual(small, 0.5)
        self.assertGreater(large, small)

    def test_regime_throttle_blocks_high_vol_non_trending(self) -> None:
        blocked = allow_signal_for_regime(
            {"reg_high_vol_state": 1.0, "reg_trending_state": 0.0},
            policy_name="high_vol_or_non_trending_off_v1",
        )
        allowed = allow_signal_for_regime(
            {"reg_high_vol_state": 0.0, "reg_trending_state": 1.0},
            policy_name="high_vol_or_non_trending_off_v1",
        )
        self.assertFalse(blocked)
        self.assertTrue(allowed)

    def test_event_backtest_applies_sizing_and_regime_throttle(self) -> None:
        records = [
            {
                "candidate_id": "a",
                "direction": "long",
                "probability": 0.80,
                "entry_time": "2025-01-02T14:32:00+00:00",
                "exit_time": "2025-01-02T14:40:00+00:00",
                "entry_price": 100.0,
                "exit_price": 101.0,
                "stop_price": 99.0,
                "pnl_r": 1.0,
                "session_date": "2025-01-02",
                "reg_high_vol_state": 0.0,
                "reg_trending_state": 1.0,
            },
            {
                "candidate_id": "b",
                "direction": "long",
                "probability": 0.70,
                "entry_time": "2025-01-02T14:45:00+00:00",
                "exit_time": "2025-01-02T14:50:00+00:00",
                "entry_price": 100.0,
                "exit_price": 99.0,
                "stop_price": 99.0,
                "pnl_r": -1.0,
                "session_date": "2025-01-02",
                "reg_high_vol_state": 1.0,
                "reg_trending_state": 0.0,
            },
        ]
        summary = run_event_driven_policy_backtest(
            records,
            threshold=0.45,
            sizing_policy="confidence_linear_v1",
            regime_throttle_policy="high_vol_or_non_trending_off_v1",
        )
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["trade_count"], 1)
        self.assertEqual(summary["throttled_signals"], 1)
        self.assertGreater(summary["avg_size_multiplier"], 0.5)


if __name__ == "__main__":
    unittest.main()
