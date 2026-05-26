import unittest

from trading_ml.model_diagnostics_lab import build_model_diagnostics_lab


class ModelDiagnosticsLabTests(unittest.TestCase):
    def test_calibration_review_is_emitted(self) -> None:
        prediction_records = [
            {
                "candidate_id": f"c{i}",
                "label": 1 if i % 2 == 0 else 0,
                "probability": 0.8 if i % 2 == 0 else 0.2,
                "pnl_r": 1.0 if i % 2 == 0 else -1.0,
            }
            for i in range(20)
        ]
        diagnostics = build_model_diagnostics_lab(prediction_records)
        self.assertEqual(diagnostics["status"], "complete")
        self.assertIn("calibration_review", diagnostics)
        self.assertIn("ece", diagnostics["calibration_review"])
        self.assertIn("reliability_rows", diagnostics["calibration_review"])


if __name__ == "__main__":
    unittest.main()
