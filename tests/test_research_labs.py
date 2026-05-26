import unittest

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.market_structure_lab import build_market_structure_lab
from trading_ml.model_diagnostics_lab import build_model_diagnostics_lab
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine


class ResearchLabsTests(unittest.TestCase):
    def test_stage2_emits_market_structure_and_model_diagnostics(self) -> None:
        state = build_agent_loop_state()
        result = run_stage2_research_engine(Stage2Config(**state["stage2_config"]))
        self.assertEqual(result["market_structure_lab"]["status"], "complete")
        self.assertIn("structure_families", result["market_structure_lab"])
        self.assertEqual(result["model_diagnostics"]["status"], "complete")
        self.assertIn("bucket_rows", result["model_diagnostics"])
        self.assertIn("shap_analysis", result["model_diagnostics"])

    def test_market_structure_lab_builds_failure_taxonomy(self) -> None:
        candidates = [
            {
                "candidate_id": "a",
                "reclaim_count": 1,
                "trace": {
                    "first_break_close_confirmed": 1.0,
                    "deepest_zone_retrace_fraction": 0.2,
                    "post_reclaim_close_strength": 0.7,
                },
            },
            {
                "candidate_id": "b",
                "reclaim_count": 0,
                "trace": {
                    "first_break_close_confirmed": 0.0,
                    "deepest_zone_retrace_fraction": 0.9,
                    "post_reclaim_close_strength": 0.1,
                },
            },
        ]
        labels = [
            {
                "candidate_id": "a",
                "label": 1,
                "outcome": "target",
                "pnl_r": 1.5,
                "bars_held": 3,
                "mfe": 2.0,
                "mae": -0.2,
            },
            {
                "candidate_id": "b",
                "label": 0,
                "outcome": "stop",
                "pnl_r": -1.0,
                "bars_held": 2,
                "mfe": 0.2,
                "mae": -1.0,
            },
        ]
        result = build_market_structure_lab(candidates, labels)
        self.assertEqual(result["status"], "complete")
        self.assertGreaterEqual(len(result["failure_taxonomy"]), 1)

    def test_model_diagnostics_lab_checks_bucket_monotonicity(self) -> None:
        records = [
            {"candidate_id": "a", "probability": 0.1, "label": 0, "pnl_r": -1.0},
            {"candidate_id": "b", "probability": 0.2, "label": 0, "pnl_r": -0.5},
            {"candidate_id": "c", "probability": 0.7, "label": 1, "pnl_r": 1.0},
            {"candidate_id": "d", "probability": 0.9, "label": 1, "pnl_r": 1.5},
        ]
        result = build_model_diagnostics_lab(records)
        self.assertEqual(result["status"], "complete")
        self.assertIn("bucket_monotonicity", result)
        self.assertIn("shap_analysis", result)


if __name__ == "__main__":
    unittest.main()
