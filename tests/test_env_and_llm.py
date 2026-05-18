import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from trading_ml.env import load_env_file
from trading_ml.llm import _normalize_openai_model_name


class EnvAndLlmTests(unittest.TestCase):
    def test_load_env_file_reads_key_values(self) -> None:
        with TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("A=1\nB=two\n", encoding="utf-8")
            loaded = load_env_file(env_path, override=True)
            self.assertEqual(loaded["A"], "1")
            self.assertEqual(os.environ["B"], "two")

    def test_model_name_normalization_strips_openai_prefix(self) -> None:
        self.assertEqual(_normalize_openai_model_name("openai:gpt-5.5"), "gpt-5.5")
        self.assertEqual(_normalize_openai_model_name("gpt-5.5"), "gpt-5.5")


if __name__ == "__main__":
    unittest.main()
