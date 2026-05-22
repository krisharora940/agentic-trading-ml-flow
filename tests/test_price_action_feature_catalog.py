import unittest

from trading_ml.price_action_feature_catalog import (
    build_catalog_feature_proposals,
    build_feature_catalog_index,
    build_strategy_intake,
    load_feature_catalog,
)


class PriceActionFeatureCatalogTests(unittest.TestCase):
    def test_load_feature_catalog_reads_groups_and_features(self) -> None:
        catalog = load_feature_catalog()
        self.assertIn("groups", catalog)
        self.assertIn("features", catalog)
        self.assertIn("auction", catalog["groups"])
        self.assertIn("opening_drive_strength", catalog["features"])

    def test_build_strategy_intake_uses_catalog_groups(self) -> None:
        intake = build_strategy_intake("Opening reclaim strength and VWAP context matter.")
        self.assertIn("auction", intake["selected_feature_groups"])
        self.assertIn("structure", intake["selected_feature_groups"])
        self.assertTrue(intake["feature_backlog"]["auction"])
        self.assertTrue(intake["feature_catalog_candidates"])
        self.assertIn("feature_catalog_groups", intake)

    def test_build_catalog_feature_proposals_uses_cluster_evidence(self) -> None:
        proposals = build_catalog_feature_proposals(
            "",
            top_cluster={
                "family": "no_follow_through",
                "recommended_family": "exit_behavior_research",
                "recommended_focus": ["followthrough_strength", "candle_tempo_decay"],
            },
        )
        self.assertIn("momentum", proposals["selected_feature_groups"])
        self.assertIn("candles", proposals["selected_feature_groups"])
        self.assertTrue(proposals["feature_catalog_candidates"])
        self.assertIn("feature_claim", proposals)

    def test_build_feature_catalog_index_matches_feature_names(self) -> None:
        index = build_feature_catalog_index()
        self.assertIn("directional_efficiency", index)
        self.assertEqual(index["directional_efficiency"]["group"], "momentum")


if __name__ == "__main__":
    unittest.main()
