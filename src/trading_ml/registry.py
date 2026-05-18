from __future__ import annotations

from dataclasses import asdict
import json

from trading_ml.paths import EXPERIMENTS_DIR
from trading_ml.schemas import ExperimentRecord


REGISTRY_PATH = EXPERIMENTS_DIR / "registry.jsonl"


def append_experiment_record(record: ExperimentRecord) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
