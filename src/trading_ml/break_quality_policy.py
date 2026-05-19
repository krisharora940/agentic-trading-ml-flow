from __future__ import annotations

from typing import Any


def get_break_quality_policies() -> list[dict[str, Any]]:
    return [
        {"name": "baseline_no_gate"},
        {"name": "break_eff_ge_0.10", "min_break_efficiency_ratio": 0.10},
        {"name": "break_eff_ge_0.20", "min_break_efficiency_ratio": 0.20},
        {"name": "break_eff_ge_0.30", "min_break_efficiency_ratio": 0.30},
        {"name": "body_ge_0.40", "min_break_body_fraction": 0.40},
        {"name": "body_ge_0.50", "min_break_body_fraction": 0.50},
        {
            "name": "balanced_deep_eff_ge_0.20",
            "apply_subtypes": ["balanced_reclaim_continuation", "deep_retrace_repair"],
            "min_break_efficiency_ratio": 0.20,
        },
        {
            "name": "balanced_deep_eff_ge_0.20_body_ge_0.40",
            "apply_subtypes": ["balanced_reclaim_continuation", "deep_retrace_repair"],
            "min_break_efficiency_ratio": 0.20,
            "min_break_body_fraction": 0.40,
        },
    ]


def get_break_quality_policy(name: str) -> dict[str, Any]:
    for policy in get_break_quality_policies():
        if policy["name"] == name:
            return policy
    raise KeyError(f"Unknown break-quality policy: {name}")


def apply_break_quality_policy(
    stitched_prediction_records: list[dict[str, Any]],
    feature_records: list[dict[str, Any]],
    *,
    policy_name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    policy = get_break_quality_policy(policy_name)
    feature_map = {str(row["candidate_id"]): row for row in feature_records}
    selected: list[dict[str, Any]] = []
    scoped_subtypes = set(policy.get("apply_subtypes", []))
    for record in stitched_prediction_records:
        if float(record.get("probability", 0.0) or 0.0) < threshold:
            continue
        feature_row = feature_map.get(str(record["candidate_id"]), {})
        merged = dict(record)
        merged.update(feature_row)
        subtype = str(merged.get("setup_subtype", "unknown"))
        in_scope = not scoped_subtypes or subtype in scoped_subtypes
        if in_scope:
            if "min_break_efficiency_ratio" in policy:
                if float(merged.get("break_efficiency_ratio", 0.0) or 0.0) < float(policy["min_break_efficiency_ratio"]):
                    continue
            if "min_break_body_fraction" in policy:
                if float(merged.get("break_body_fraction", 0.0) or 0.0) < float(policy["min_break_body_fraction"]):
                    continue
        selected.append(merged)
    return selected
