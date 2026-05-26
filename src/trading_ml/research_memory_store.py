from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trading_ml.paths import REPORTS_DIR


MEMORY_DIR = REPORTS_DIR / "memory"
FAILURE_MEMORY_PATH = MEMORY_DIR / "failure_memory.jsonl"
ACTION_HISTORY_PATH = MEMORY_DIR / "research_action_history.jsonl"
DESK_MEMORY_PATH = MEMORY_DIR / "desk_memory.jsonl"


def load_persisted_research_memory() -> dict[str, list[dict[str, Any]]]:
    return {
        "failure_memory": _dedupe_by_signature(
            _read_jsonl(FAILURE_MEMORY_PATH),
            keys=("family", "hypothesis_id", "failure_type", "status"),
        ),
        "research_action_history": _dedupe_by_signature(
            _read_jsonl(ACTION_HISTORY_PATH),
            keys=("action_id", "family", "hypothesis_id", "proposal_id"),
        ),
        "desk_memory": _dedupe_by_signature(
            _read_jsonl(DESK_MEMORY_PATH), keys=("proposal_id",)
        ),
    }


def append_failure_memory_entry(entry: dict[str, Any]) -> None:
    _append_jsonl(FAILURE_MEMORY_PATH, entry)


def append_action_history_entry(entry: dict[str, Any]) -> None:
    _append_jsonl(ACTION_HISTORY_PATH, entry)


def append_desk_memory_entry(entry: dict[str, Any]) -> None:
    _append_jsonl(DESK_MEMORY_PATH, entry)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _dedupe_by_signature(
    rows: list[dict[str, Any]], *, keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        signature = tuple(str(row.get(key, "")) for key in keys)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(row)
    return deduped[-100:]
