from __future__ import annotations

import os
from typing import Any

from trading_ml.env import load_runtime_env


def get_default_model_name() -> str:
    load_runtime_env()
    return os.environ.get("TRADING_ML_MODEL", "openai:gpt-5.5")


def _normalize_openai_model_name(model_name: str) -> str:
    if model_name.startswith("openai:"):
        return model_name.split(":", 1)[1]
    return model_name


def create_chat_model(model_name: str | None = None, **kwargs: Any) -> Any:
    load_runtime_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the runtime environment.")

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "langchain-openai is not installed. Install the optional agents dependencies with "
            "`pip install -e .[agents]`."
        ) from exc

    resolved_model = _normalize_openai_model_name(model_name or get_default_model_name())
    return ChatOpenAI(model=resolved_model, api_key=api_key, **kwargs)


def llm_enabled() -> bool:
    load_runtime_env()
    return bool(os.environ.get("OPENAI_API_KEY"))
