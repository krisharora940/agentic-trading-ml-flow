from __future__ import annotations

from pathlib import Path
import json
import tomllib
from typing import Any

from trading_ml.paths import CONFIGS_DIR, MANIFESTS_DIR


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_global_config() -> dict[str, Any]:
    return load_toml(CONFIGS_DIR / "global.toml")


def load_evidence_boundary_config() -> dict[str, Any]:
    return load_toml(CONFIGS_DIR / "evidence_boundary.toml")


def load_skill_registry_config() -> dict[str, Any]:
    return load_toml(CONFIGS_DIR / "skills.toml")


def load_bnr_config() -> dict[str, Any]:
    return load_toml(CONFIGS_DIR / "bnr.toml")


def load_research_program_config() -> dict[str, Any]:
    return load_toml(CONFIGS_DIR / "research_program.toml")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_databento_manifest(name: str = "databento_mnq_manifest.json") -> dict[str, Any]:
    return load_json(MANIFESTS_DIR / name)
