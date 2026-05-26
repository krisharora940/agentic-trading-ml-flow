import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "run_cpcv_failure_attribution.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_cpcv_failure_attribution", MODULE_PATH
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CpcvFailureAttributionTests(unittest.TestCase):
    def test_returns_insufficient_artifacts_when_rows_missing(self) -> None:
        payload = {
            "cpcv": {
                "status": "fail",
                "pbo": 0.36,
                "mean_total_pnl_r": -0.49,
                "median_total_pnl_r": 1.86,
                "min_path_pnl_r": -19.89,
                "path_positive_rate": 0.63,
                "distribution": {"p10_total_pnl_r": -10.0},
                "worst_paths": [{"path_id": "cpcv_001"}],
                "best_paths": [{"path_id": "cpcv_002"}],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "exploration_benchmark_diagnostics.json"
            report.write_text(json.dumps(payload), encoding="utf-8")
            result = MODULE.build_cpcv_failure_attribution(report)

        self.assertEqual(result["status"], "insufficient_artifacts")
        self.assertEqual(result["reason"], "missing_persisted_cpcv_rows")
        self.assertEqual(result["cpcv_status"], "fail")
        self.assertEqual(
            result["failure_summary"]["failure_type"], "tail_path_fragility"
        )


if __name__ == "__main__":
    unittest.main()
