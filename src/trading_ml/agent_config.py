from __future__ import annotations

from typing import Any

from trading_ml.config import load_toml
from trading_ml.paths import CONFIGS_DIR


def load_agent_loop_config() -> dict[str, Any]:
    return load_toml(CONFIGS_DIR / "agent_loop.toml")
