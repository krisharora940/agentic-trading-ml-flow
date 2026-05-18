import unittest

from trading_ml.config import (
    load_bnr_config,
    load_evidence_boundary_config,
    load_global_config,
    load_skill_registry_config,
)


class ConfigTests(unittest.TestCase):
    def test_global_config_contains_bnr_project_fields(self) -> None:
        config = load_global_config()
        self.assertEqual(config["project"]["primary_setup"], "BNR")
        self.assertEqual(config["project"]["timezone"], "America/New_York")

    def test_evidence_boundary_has_three_windows(self) -> None:
        boundary = load_evidence_boundary_config()
        self.assertEqual(boundary["boundary"]["mode"], "strict")
        self.assertTrue({"boundary", "exploration", "validation", "holdout"} <= set(boundary))

    def test_skill_registry_has_expected_roles(self) -> None:
        skills = load_skill_registry_config()
        self.assertEqual(skills["orchestrators"]["governor"], "trading-ml-governor")
        self.assertIn("ml4t-databento", skills["data"]["primary"])

    def test_bnr_config_enables_hybrid_engineer_features(self) -> None:
        config = load_bnr_config()
        self.assertTrue(config["engineer_features"]["enabled"])
        self.assertEqual(config["engineer_features"]["backend"], "hybrid")
        self.assertIn("rsi", config["engineer_features"]["features"])
        self.assertIn("translation_contract", config)
        self.assertIn("model_search_v1", config)


if __name__ == "__main__":
    unittest.main()
