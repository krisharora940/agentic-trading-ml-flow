from __future__ import annotations

import os
from typing import Any

from trading_ml.env import load_runtime_env


def get_default_model_name() -> str:
    load_runtime_env()
    return os.environ.get("TRADING_ML_MODEL", "ollama:qwen2.5:14b-instruct")


def _resolve_provider(model_name: str | None = None) -> str:
    load_runtime_env()
    provider = str(os.environ.get("TRADING_ML_LLM_PROVIDER", "") or "").strip().lower()
    resolved = str(model_name or get_default_model_name())
    if provider:
        return provider
    if resolved.startswith("openai:"):
        return "openai"
    if resolved.startswith("ollama:"):
        return "ollama"
    if resolved.startswith("openai_compatible:"):
        return "openai_compatible"
    if os.environ.get("TRADING_ML_LLM_BASE_URL"):
        return "openai_compatible"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


def _normalize_model_name(model_name: str) -> str:
    for prefix in ("openai:", "ollama:", "openai_compatible:"):
        if model_name.startswith(prefix):
            return model_name.split(":", 1)[1]
    return model_name


def _default_timeout_seconds() -> float:
    raw = str(os.environ.get("TRADING_ML_LLM_TIMEOUT_SECONDS", "45") or "45").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 45.0


def _default_max_retries() -> int:
    raw = str(os.environ.get("TRADING_ML_LLM_MAX_RETRIES", "0") or "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def create_chat_model(model_name: str | None = None, **kwargs: Any) -> Any:
    load_runtime_env()
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "langchain-openai is not installed. Install the optional agents dependencies with "
            "`pip install -e .[agents]`."
        ) from exc

    resolved_model = _normalize_model_name(model_name or get_default_model_name())
    provider = _resolve_provider(model_name)
    request_kwargs = {
        "timeout": _default_timeout_seconds(),
        "max_retries": _default_max_retries(),
        **kwargs,
    }
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the runtime environment.")
        return ChatOpenAI(model=resolved_model, api_key=api_key, **request_kwargs)
    if provider == "ollama":
        base_url = os.environ.get("TRADING_ML_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        api_key = os.environ.get("TRADING_ML_LLM_API_KEY", "ollama")
        return ChatOpenAI(model=resolved_model, api_key=api_key, base_url=base_url, **request_kwargs)
    if provider == "openai_compatible":
        base_url = os.environ.get("TRADING_ML_LLM_BASE_URL")
        if not base_url:
            raise RuntimeError("TRADING_ML_LLM_BASE_URL is not set for openai_compatible LLM usage.")
        api_key = os.environ.get("TRADING_ML_LLM_API_KEY", "local")
        return ChatOpenAI(model=resolved_model, api_key=api_key, base_url=base_url, **request_kwargs)
    raise RuntimeError(f"Unsupported LLM provider: {provider}")


def llm_enabled() -> bool:
    load_runtime_env()
    provider = _resolve_provider()
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "ollama":
        return True
    if provider == "openai_compatible":
        return bool(os.environ.get("TRADING_ML_LLM_BASE_URL"))
    return False
