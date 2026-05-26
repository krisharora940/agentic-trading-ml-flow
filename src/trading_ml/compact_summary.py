from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_KEEP_KEYS = {
    "run_id",
    "current_node",
    "research_cycle",
    "phase",
    "status",
    "decision",
    "proposal_id",
    "proposal_family",
    "family",
    "action_id",
    "plan_id",
    "result_id",
    "selected_node",
    "batch_decision",
    "trial_count",
    "blocking_issues",
    "blocked_reasons",
}


def compact_summary(
    value: Any,
    *,
    max_depth: int = 3,
    max_items: int = 5,
    keep_keys: set[str] | None = None,
) -> Any:
    keys = keep_keys or DEFAULT_KEEP_KEYS
    return _compact(
        value, depth=0, max_depth=max_depth, max_items=max_items, keep_keys=keys
    )


def compact_graph_state(state: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: state.get(key)
        for key in (
            "run_id",
            "current_node",
            "research_cycle",
            "phase",
            "search_batch_status",
            "promotion_decision",
            "blocking_issues",
        )
        if key in state
    }
    desk_summary = dict(state.get("desk_summary", {}) or {})
    if desk_summary:
        summary["desk_summary"] = compact_summary(desk_summary, max_depth=2)
    for key in (
        "desk_proposals",
        "research_action_plan",
        "red_team_review",
        "research_action_result",
        "marginal_evidence",
        "state_ontology",
        "research_branch_status",
        "responsibility_boundaries",
    ):
        if key in state:
            summary[key] = compact_summary(state[key], max_depth=2)
    return summary


def _compact(
    value: Any,
    *,
    depth: int,
    max_depth: int,
    max_items: int,
    keep_keys: set[str],
) -> Any:
    if depth >= max_depth:
        return _shape(value)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in list(value.items())[: max_items * 2]:
            key_str = str(key)
            if key_str in keep_keys or depth < max_depth - 1:
                output[key_str] = _compact(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    keep_keys=keep_keys,
                )
            if len(output) >= max_items:
                break
        hidden = max(0, len(value) - len(output))
        if hidden:
            output["_omitted_keys"] = hidden
        return output
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        rows = [
            _compact(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                keep_keys=keep_keys,
            )
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            rows.append({"_omitted_items": len(value) - max_items})
        return rows
    return value


def _shape(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            "_type": "dict",
            "keys": list(value.keys())[:8],
            "key_count": len(value),
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return {"_type": "list", "count": len(value)}
    return value
