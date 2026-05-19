from __future__ import annotations

from typing import Any


def apply_subtype_policy(
    prediction_records: list[dict[str, Any]],
    *,
    include_subtypes: list[str] | None = None,
    exclude_subtypes: list[str] | None = None,
    default_threshold: float = 0.45,
    threshold_overrides: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    include = set(include_subtypes or [])
    exclude = set(exclude_subtypes or [])
    overrides = dict(threshold_overrides or {})
    selected: list[dict[str, Any]] = []

    for record in prediction_records:
        subtype = str(record.get("setup_subtype", "unknown"))
        probability = float(record.get("probability", 0.0) or 0.0)
        if include and subtype not in include:
            continue
        if subtype in exclude:
            continue
        threshold = float(overrides.get(subtype, default_threshold))
        if probability >= threshold:
            enriched = dict(record)
            enriched["applied_threshold"] = threshold
            selected.append(enriched)
    return selected


def summarize_policy_records(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        subtype = str(record.get("setup_subtype", "unknown"))
        counts[subtype] = counts.get(subtype, 0) + 1
    return counts
