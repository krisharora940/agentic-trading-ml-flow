import os
import unittest
from unittest import mock

from trading_ml.engineer_adapter import engineer_runtime_enabled


class EngineerAdapterTests(unittest.TestCase):
    def test_ml4t_engineer_backend_is_opt_in_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRADING_ML_ENABLE_ML4T_ENGINEER", None)
            self.assertFalse(engineer_runtime_enabled())


if __name__ == "__main__":
    unittest.main()
