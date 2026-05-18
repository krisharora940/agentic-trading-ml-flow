from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from trading_ml.paths import LOGS_DIR
from trading_ml.schemas import utc_now_iso


@dataclass(slots=True)
class RunLogEvent:
    event_type: str
    actor: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


class JsonlRunLogger:
    def __init__(self, run_id: str, log_dir: Path | None = None) -> None:
        self.run_id = run_id
        self.log_dir = Path(log_dir) if log_dir is not None else LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{run_id}.jsonl"

    def log(self, event: RunLogEvent) -> Path:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        return self.path
