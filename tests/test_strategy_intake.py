import unittest

from trading_ml.price_action_feature_catalog import build_strategy_intake


class StrategyIntakeTests(unittest.TestCase):
    def test_strategy_intake_extracts_feature_groups_from_notes(self) -> None:
        intake = build_strategy_intake(
            "The setup depends on reclaim quality, break quality, VWAP context, and opening volume."
        )
        groups = set(intake["selected_feature_groups"])
        self.assertIn("structure", groups)
        self.assertIn("momentum", groups)
        self.assertIn("auction", groups)
        self.assertIn("liquidity", groups)
        self.assertTrue(intake["feature_catalog_candidates"])


if __name__ == "__main__":
    unittest.main()
