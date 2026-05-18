import unittest

from trading_ml.bootstrap import build_initial_project_state
from trading_ml.schemas import ExperimentRecord


class SchemaTests(unittest.TestCase):
    def test_initial_project_state_starts_in_foundation_phase(self) -> None:
        state = build_initial_project_state()
        self.assertEqual(state.phase, "foundation")
        self.assertIsNotNone(state.bnr_spec)
        self.assertEqual(state.evidence_boundary.mode, "strict")

    def test_experiment_record_serializes_expected_fields(self) -> None:
        record = ExperimentRecord(
            experiment_id="exp-001",
            hypothesis="Opening zone break continuation improves expectancy.",
            config_ref="configs/global.toml",
            data_slice={"start": "2024-01-01", "end": "2024-06-30"},
            result={"sharpe": 0.0},
            decision="queued",
            phase="exploration",
        )
        self.assertEqual(record.experiment_id, "exp-001")
        self.assertEqual(record.phase, "exploration")


if __name__ == "__main__":
    unittest.main()
