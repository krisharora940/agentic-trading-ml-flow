import unittest

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.translation_analysis import build_translation_analysis
from trading_ml.validation_audit import build_validation_audit


class ModelingAndTranslationTests(unittest.TestCase):
    def test_stage2_supports_gbm_and_translation_outputs(self) -> None:
        state = build_agent_loop_state()
        config = dict(state["stage2_config"])
        config["model_family"] = "gbm"
        result = run_stage2_research_engine(Stage2Config(**config))
        model_summary = result["model_summary"]
        self.assertEqual(model_summary["model_family"], "gbm")
        self.assertGreaterEqual(len(model_summary["prediction_records"]), 1)
        translation = build_translation_analysis(result)
        self.assertIn(translation["status"], {"pass", "fail", "pending"})
        self.assertIn("rows", translation)

    def test_walk_forward_stitched_translation_outputs(self) -> None:
        state = build_agent_loop_state()
        result = run_stage2_research_engine(Stage2Config(**state["stage2_config"]))
        validation = build_validation_audit(result, {})
        stitched = validation["walk_forward"].get("stitched_prediction_records", [])
        self.assertGreaterEqual(len(stitched), 1)
        translation = build_translation_analysis(result, prediction_records=stitched)
        self.assertIn(translation["status"], {"pass", "fail", "pending"})
        self.assertIn("best_threshold", translation)
        self.assertIn("rows", translation)
        execution = run_event_driven_policy_backtest(stitched, threshold=0.6)
        self.assertIn(execution["status"], {"complete", "pending"})
        if execution["status"] == "complete":
            self.assertGreaterEqual(execution["trade_count"], 1)
            self.assertIn("max_drawdown_r", execution)
            self.assertIn("session_rows", execution)
            self.assertIn("overlap_skips", execution)
            self.assertIn("fill_assumptions", execution)
            self.assertIn(
                execution["fill_assumptions"]["profile"], {"base", "stressed"}
            )
            self.assertIn("sizing_policy", execution["fill_assumptions"])
            self.assertIn("regime_throttle_policy", execution["fill_assumptions"])


if __name__ == "__main__":
    unittest.main()
