import unittest

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.translation_analysis import build_translation_analysis


class ModelingAndTranslationTests(unittest.TestCase):
    def test_stage2_supports_gbm_and_translation_outputs(self) -> None:
        state = build_agent_loop_state()
        result = run_stage2_research_engine(Stage2Config(**state["stage2_config"], model_family="gbm"))
        model_summary = result["model_summary"]
        self.assertEqual(model_summary["model_family"], "gbm")
        self.assertGreaterEqual(len(model_summary["prediction_records"]), 1)
        translation = build_translation_analysis(result)
        self.assertIn(translation["status"], {"pass", "fail", "pending"})
        self.assertIn("rows", translation)


if __name__ == "__main__":
    unittest.main()
