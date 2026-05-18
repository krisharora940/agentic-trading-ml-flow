from __future__ import annotations

from dataclasses import dataclass

from trading_ml.config import load_skill_registry_config


@dataclass(frozen=True, slots=True)
class SkillRegistry:
    raw: dict

    @classmethod
    def load(cls) -> "SkillRegistry":
        return cls(raw=load_skill_registry_config())

    def get(self, group: str, key: str = "primary") -> list[str] | str:
        return self.raw[group][key]
