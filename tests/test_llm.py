import os
import unittest
from unittest import mock

from trading_ml.llm import create_chat_model, get_default_model_name, llm_enabled


class LLMTests(unittest.TestCase):
    def test_default_model_uses_explicit_override(self) -> None:
        with mock.patch.dict(
            os.environ, {"TRADING_ML_MODEL": "ollama:qwen2.5:14b-instruct"}, clear=False
        ):
            self.assertEqual(get_default_model_name(), "ollama:qwen2.5:14b-instruct")

    def test_llm_enabled_defaults_true_for_ollama_provider(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(llm_enabled())

    def test_create_chat_model_uses_ollama_compatible_defaults(self) -> None:
        fake = object()
        env = {
            "TRADING_ML_LLM_PROVIDER": "ollama",
            "TRADING_ML_MODEL": "ollama:qwen2.5:14b-instruct",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch(
                "langchain_openai.ChatOpenAI", return_value=fake
            ) as chat_model:
                result = create_chat_model(temperature=0)
        self.assertIs(result, fake)
        chat_model.assert_called_once_with(
            model="qwen2.5:14b-instruct",
            api_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
            timeout=45.0,
            max_retries=0,
            temperature=0,
        )

    def test_create_chat_model_supports_openai_compatible_local_endpoint(self) -> None:
        fake = object()
        env = {
            "TRADING_ML_LLM_PROVIDER": "openai_compatible",
            "TRADING_ML_LLM_BASE_URL": "http://127.0.0.1:1234/v1",
            "TRADING_ML_MODEL": "openai_compatible:qwen/qwen3-14b",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch(
                "langchain_openai.ChatOpenAI", return_value=fake
            ) as chat_model:
                result = create_chat_model(temperature=0)
        self.assertIs(result, fake)
        chat_model.assert_called_once_with(
            model="qwen/qwen3-14b",
            api_key="local",
            base_url="http://127.0.0.1:1234/v1",
            timeout=45.0,
            max_retries=0,
            temperature=0,
        )

    def test_create_chat_model_honors_timeout_and_retry_overrides(self) -> None:
        fake = object()
        env = {
            "TRADING_ML_LLM_PROVIDER": "ollama",
            "TRADING_ML_MODEL": "ollama:qwen2.5:14b-instruct",
            "TRADING_ML_LLM_TIMEOUT_SECONDS": "12",
            "TRADING_ML_LLM_MAX_RETRIES": "2",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch(
                "langchain_openai.ChatOpenAI", return_value=fake
            ) as chat_model:
                result = create_chat_model(temperature=0)
        self.assertIs(result, fake)
        chat_model.assert_called_once_with(
            model="qwen2.5:14b-instruct",
            api_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
            timeout=12.0,
            max_retries=2,
            temperature=0,
        )


if __name__ == "__main__":
    unittest.main()
