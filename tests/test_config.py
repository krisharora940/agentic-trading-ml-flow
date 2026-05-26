import unittest

from trading_ml.config import (
    load_bnr_config,
    load_evidence_boundary_config,
    load_global_config,
    load_research_program_config,
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
        self.assertTrue(
            {"boundary", "exploration", "validation", "holdout"} <= set(boundary)
        )

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
        self.assertIn("utility_contract", config)
        self.assertIn("model_search_v1", config)
        self.assertIn("feature_threshold_search_v1", config)
        self.assertIn("label_search_v1", config)
        self.assertIn("threshold_search_v1", config)
        self.assertIn("translation_policy_search_v1", config)
        self.assertEqual(
            config["frozen_benchmark"]["feature_family"], "bnr_plus_context"
        )
        self.assertIn("sizing_policy", config["frozen_benchmark"])
        self.assertIn("regime_throttle_policy", config["frozen_benchmark"])
        self.assertIn("regime_size_policy", config["frozen_benchmark"])

    def test_research_program_config_defines_institutional_workstreams(self) -> None:
        config = load_research_program_config()
        self.assertEqual(
            config["program"]["primary_objective"],
            "maximize_utility_subject_to_research_validity",
        )
        self.assertIn("thesis_lab", config["program"]["workstreams"])
        self.assertIn("execution_lab", config["program"]["workstreams"])
        self.assertEqual(
            config["program"]["domain_research"]["priority_sources"][0], "ml4trading.io"
        )


if __name__ == "__main__":
    unittest.main()
