from __future__ import annotations

from statistics import median
from typing import Any

from trading_ml.schemas import (
    ContinuationProfile,
    FailureProfile,
    StateOntology,
    StateTransition,
    utc_now_iso,
)


PRIMARY_MODELING_TARGET = "auction_state_continuation_validity"
BNR_ROLE = "event_trigger_within_state_machine"


def build_state_ontology(
    attempts: list[dict[str, Any]],
    failure_clusters: list[dict[str, Any]],
) -> dict[str, Any]:
    state_definitions = _state_definitions(attempts)
    transitions = _state_transitions(attempts)
    continuation_profiles = _continuation_profiles(attempts)
    failure_profiles = _failure_profiles(failure_clusters)
    ontology = StateOntology(
        ontology_id=f"STATE-ONT-{utc_now_iso()}",
        version=1,
        primary_modeling_target=PRIMARY_MODELING_TARGET,
        bnr_role=BNR_ROLE,
        state_definitions=state_definitions,
        transitions=transitions,
        continuation_profiles=continuation_profiles,
        failure_profiles=failure_profiles,
        persistence_statistics=_persistence_statistics(attempts),
    )
    return ontology.to_dict()


def _state_definitions(attempts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for key in ("environment_state", "setup_state", "path_class"):
        counts = _counts(attempts, key)
        for value, count in counts.items():
            state_key = f"{key}:{value}"
            definitions[state_key] = {
                "axis": key,
                "label": value,
                "sample_size": count,
                "definition_status": "empirical_observed",
            }
    if not definitions:
        definitions["auction_state:unknown"] = {
            "axis": "auction_state",
            "label": "unknown",
            "sample_size": 0,
            "definition_status": "pending_observation",
        }
    return definitions


def _state_transitions(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in attempts:
        from_state = str(row.get("environment_state", "unknown") or "unknown")
        to_state = str(row.get("path_class", "unknown") or "unknown")
        grouped.setdefault((from_state, to_state), []).append(row)
    totals = _counts(attempts, "environment_state")
    transitions: list[dict[str, Any]] = []
    for (from_state, to_state), rows in sorted(grouped.items()):
        total = max(int(totals.get(from_state, len(rows)) or len(rows)), 1)
        transitions.append(
            StateTransition(
                from_state=from_state,
                to_state=to_state,
                trigger="bnr_event",
                sample_size=len(rows),
                transition_probability=len(rows) / total,
                persistence_bars=_median_numeric(rows, "bars_held"),
                evidence={"event_role": BNR_ROLE},
            ).to_dict()
        )
    return transitions


def _continuation_profiles(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in attempts:
        grouped.setdefault(str(row.get("environment_state", "unknown")), []).append(row)
    profiles: list[dict[str, Any]] = []
    for state, rows in sorted(grouped.items()):
        positives = sum(1 for row in rows if _is_continuation(row))
        failures = sum(1 for row in rows if _is_failure(row))
        profiles.append(
            ContinuationProfile(
                state=state,
                sample_size=len(rows),
                continuation_rate=positives / max(len(rows), 1),
                failure_rate=failures / max(len(rows), 1),
                avg_pnl_r=_avg_numeric(rows, "pnl_r"),
                median_persistence_bars=_median_numeric(rows, "bars_held"),
                evidence={
                    "primary_modeling_target": PRIMARY_MODELING_TARGET,
                    "bnr_role": BNR_ROLE,
                },
            ).to_dict()
        )
    return profiles


def _failure_profiles(failure_clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for cluster in failure_clusters:
        evidence = dict(cluster.get("evidence", {}) or {})
        profiles.append(
            FailureProfile(
                failure_family=str(cluster.get("family", "unknown")),
                state=str(cluster.get("dominant_environment_state", "unknown")),
                sample_size=int(cluster.get("rows", 0) or 0),
                avg_pnl_r=_float_or_none(cluster.get("avg_pnl_r")),
                dominant_path_class=str(evidence.get("path_class_mode", "unknown")),
                dominant_repair_state=str(
                    cluster.get("dominant_setup_state", "unknown")
                ),
                evidence=evidence,
            ).to_dict()
        )
    return profiles


def _persistence_statistics(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "attempt_count": len(attempts),
        "environment_state_counts": _counts(attempts, "environment_state"),
        "path_class_counts": _counts(attempts, "path_class"),
        "median_bars_held": _median_numeric(attempts, "bars_held"),
    }


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown") or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _is_continuation(row: dict[str, Any]) -> bool:
    return (
        str(row.get("path_class", "")).lower() in {"runner", "continuation"}
        or str(row.get("outcome", "")).lower() == "target"
        or row.get("label") == 1
    )


def _is_failure(row: dict[str, Any]) -> bool:
    return (
        str(row.get("path_class", "")).lower() in {"failure", "chop"}
        or str(row.get("outcome", "")).lower() in {"stop", "timeout"}
        or row.get("label") == 0
    )


def _avg_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _median_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_float_or_none(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(median(clean))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
