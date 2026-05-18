from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class WalkForwardFold:
    fold: int
    train_sessions: list[str]
    test_sessions: list[str]
    train_rows: int
    test_rows: int
    purged_rows: int
    embargo_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
