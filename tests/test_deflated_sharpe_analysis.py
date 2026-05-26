import unittest

from trading_ml.deflated_sharpe_analysis import (
    compute_sharpe_ratio,
    deflated_sharpe_probability,
)


class DeflatedSharpeTests(unittest.TestCase):
    def test_compute_sharpe_ratio_positive_for_positive_returns(self) -> None:
        returns = [0.5, 0.2, 0.4, 0.1, 0.3]
        sharpe = compute_sharpe_ratio(returns)
        self.assertIsNotNone(sharpe)
        self.assertGreater(sharpe, 0)

    def test_deflated_sharpe_probability_penalizes_many_trials(self) -> None:
        low_trials = deflated_sharpe_probability(
            observed_sr=1.5,
            n_trials=5,
            sr_std=0.4,
            n_obs=50,
            skew=0.0,
            kurtosis=3.0,
        )
        high_trials = deflated_sharpe_probability(
            observed_sr=1.5,
            n_trials=100,
            sr_std=0.4,
            n_obs=50,
            skew=0.0,
            kurtosis=3.0,
        )
        self.assertGreater(low_trials, high_trials)


if __name__ == "__main__":
    unittest.main()
