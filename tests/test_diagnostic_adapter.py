import os
import unittest
from unittest import mock

from trading_ml.diagnostic_adapter import diagnostic_available


class DiagnosticAdapterTests(unittest.TestCase):
    def test_ml4t_diagnostic_backend_is_opt_in_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRADING_ML_ENABLE_ML4T_DIAGNOSTIC", None)
            self.assertFalse(diagnostic_available())


if __name__ == "__main__":
    unittest.main()
