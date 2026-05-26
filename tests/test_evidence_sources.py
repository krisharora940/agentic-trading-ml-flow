import unittest

from trading_ml.evidence_sources import select_manifest_source_path


class EvidenceSourceTests(unittest.TestCase):
    def test_select_manifest_source_prefers_boundary_role(self) -> None:
        manifest = {
            "files": [
                {
                    "timeframe": "30s",
                    "source_path": "all.parquet",
                    "stage2_priority": 100,
                },
                {
                    "timeframe": "30s",
                    "source_path": "exploration.parquet",
                    "boundary_role": "exploration",
                    "stage2_priority": 50,
                },
                {
                    "timeframe": "30s",
                    "source_path": "validation.parquet",
                    "boundary_role": "validation",
                    "stage2_priority": 200,
                },
            ]
        }
        self.assertEqual(
            select_manifest_source_path(
                manifest, timeframe="30s", boundary_role="exploration"
            ),
            "exploration.parquet",
        )


if __name__ == "__main__":
    unittest.main()
