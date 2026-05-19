from __future__ import annotations

from typing import Any


def select_manifest_source_path(
    manifest: dict[str, Any],
    *,
    timeframe: str = "30s",
    boundary_role: str | None = None,
) -> str | None:
    files = manifest.get("files", [])
    candidates = [entry for entry in files if entry.get("timeframe") == timeframe]
    if boundary_role:
        scoped = [entry for entry in candidates if entry.get("boundary_role") == boundary_role]
        if scoped:
            candidates = scoped
    if not candidates:
        return None
    chosen = max(
        candidates,
        key=lambda entry: (
            entry.get("stage2_priority", 0),
            entry.get("sessions", 0),
            entry.get("rows", 0),
        ),
    )
    return chosen.get("source_path")
