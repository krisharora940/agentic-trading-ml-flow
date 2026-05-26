from __future__ import annotations

import os
from pathlib import Path

from trading_ml.paths import ROOT


def load_env_file(
    path: Path | None = None, *, override: bool = False
) -> dict[str, str]:
    env_path = path or (ROOT / ".env")
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def load_runtime_env(*, override: bool = False) -> dict[str, str]:
    return load_env_file(override=override)
