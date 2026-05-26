from __future__ import annotations

from typing import Any


def get_reclaim_meta_policies() -> list[dict[str, Any]]:
    return [
        {"name": "baseline_gated"},
        {
            "name": "balanced_clean_reclaim_close_ge_0.15",
            "apply_subtypes": [
                "balanced_reclaim_continuation",
                "clean_break_continuation",
            ],
            "min_reclaim_close_location": 0.15,
        },
        {
            "name": "balanced_clean_reclaim_close_ge_0.25",
            "apply_subtypes": [
                "balanced_reclaim_continuation",
                "clean_break_continuation",
            ],
            "min_reclaim_close_location": 0.25,
        },
        {
            "name": "balanced_clean_post_reclaim_ge_0.30",
            "apply_subtypes": [
                "balanced_reclaim_continuation",
                "clean_break_continuation",
            ],
            "min_post_reclaim_close_strength": 0.30,
        },
        {
            "name": "balanced_clean_post_reclaim_ge_0.50",
            "apply_subtypes": [
                "balanced_reclaim_continuation",
                "clean_break_continuation",
            ],
            "min_post_reclaim_close_strength": 0.50,
        },
        {
            "name": "balanced_clean_reclaim_body_ge_0.35",
            "apply_subtypes": [
                "balanced_reclaim_continuation",
                "clean_break_continuation",
            ],
            "min_reclaim_body_strength": 0.35,
        },
        {
            "name": "balanced_clean_close_ge_0.15_post_ge_0.30",
            "apply_subtypes": [
                "balanced_reclaim_continuation",
                "clean_break_continuation",
            ],
            "min_reclaim_close_location": 0.15,
            "min_post_reclaim_close_strength": 0.30,
        },
    ]


def apply_reclaim_meta_policy(
    records: list[dict[str, Any]], *, policy_name: str
) -> list[dict[str, Any]]:
    policy = next(
        policy
        for policy in get_reclaim_meta_policies()
        if policy["name"] == policy_name
    )
    scoped_subtypes = set(policy.get("apply_subtypes", []))
    selected: list[dict[str, Any]] = []
    for record in records:
        subtype = str(record.get("setup_subtype", "unknown"))
        in_scope = not scoped_subtypes or subtype in scoped_subtypes
        if in_scope:
            if "min_reclaim_close_location" in policy:
                if float(record.get("reclaim_close_location", 0.0) or 0.0) < float(
                    policy["min_reclaim_close_location"]
                ):
                    continue
            if "min_post_reclaim_close_strength" in policy:
                if float(record.get("post_reclaim_close_strength", 0.0) or 0.0) < float(
                    policy["min_post_reclaim_close_strength"]
                ):
                    continue
            if "min_reclaim_body_strength" in policy:
                if float(record.get("reclaim_body_strength", 0.0) or 0.0) < float(
                    policy["min_reclaim_body_strength"]
                ):
                    continue
        selected.append(record)
    return selected
