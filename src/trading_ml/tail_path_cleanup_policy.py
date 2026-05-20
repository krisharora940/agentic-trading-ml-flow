from __future__ import annotations

from datetime import datetime
from typing import Any


def get_tail_path_cleanup_policies() -> list[dict[str, Any]]:
    return [
        {
            "name": "exclude_deep_retrace",
            "kind": "subtype_exclusion",
            "exclude_subtypes": ["deep_retrace_repair"],
        },
        {
            "name": "deep_retrace_threshold_070",
            "kind": "subtype_threshold_lift",
            "subtype_thresholds": {"deep_retrace_repair": 0.70},
        },
        {
            "name": "deep_retrace_size_half",
            "kind": "subtype_size_haircut",
            "subtype_size_haircuts": {"deep_retrace_repair": 0.50},
        },
        {
            "name": "exclude_1432_bucket",
            "kind": "time_bucket_exclusion",
            "executable": False,
            "governance_reason": "Exact time buckets are diagnostic only; using them as gates is path overfit.",
            "exclude_time_buckets": ["14:32"],
        },
        {
            "name": "haircut_1432_bucket",
            "kind": "time_bucket_size_haircut",
            "executable": False,
            "governance_reason": "Exact time buckets are diagnostic only; using them as sizing knobs is path overfit.",
            "time_bucket_size_haircuts": {"14:32": 0.50},
        },
        {
            "name": "deep_retrace_high_conf_cap",
            "kind": "high_conf_calibration_cap",
            "subtype_probability_cap": {"deep_retrace_repair": {"min_probability": 0.65, "cap_probability": 0.64}},
        },
        {
            "name": "deep_retrace_regime_confirmed_only",
            "kind": "regime_confirmation_gate",
            "require_regime_confirmation_for_subtypes": ["deep_retrace_repair"],
        },
    ]


def apply_tail_path_cleanup_policy(
    records: list[dict[str, Any]],
    *,
    policy_name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    policy = next((row for row in get_tail_path_cleanup_policies() if row["name"] == policy_name), None)
    if policy is None:
        raise KeyError(f"Unknown tail path cleanup policy: {policy_name}")
    if policy.get("executable") is False:
        raise ValueError(f"Tail cleanup policy is diagnostic-only and cannot be executed: {policy_name}")

    filtered: list[dict[str, Any]] = []
    for row in records:
        updated = dict(row)
        updated.setdefault("policy_size_multiplier", 1.0)
        subtype = str(updated.get("setup_subtype", ""))
        time_bucket = _time_bucket(updated)
        probability = float(updated.get("probability", 0.0) or 0.0)

        if subtype in set(policy.get("exclude_subtypes", [])):
            continue
        if time_bucket in set(policy.get("exclude_time_buckets", [])):
            continue

        subtype_thresholds = dict(policy.get("subtype_thresholds", {}))
        if subtype in subtype_thresholds and probability < float(subtype_thresholds[subtype]):
            continue

        subtype_size_haircuts = dict(policy.get("subtype_size_haircuts", {}))
        if subtype in subtype_size_haircuts:
            updated["policy_size_multiplier"] = float(updated.get("policy_size_multiplier", 1.0) or 1.0) * float(subtype_size_haircuts[subtype])

        time_bucket_haircuts = dict(policy.get("time_bucket_size_haircuts", {}))
        if time_bucket in time_bucket_haircuts:
            updated["policy_size_multiplier"] = float(updated.get("policy_size_multiplier", 1.0) or 1.0) * float(time_bucket_haircuts[time_bucket])

        subtype_probability_cap = dict(policy.get("subtype_probability_cap", {}))
        cap_config = dict(subtype_probability_cap.get(subtype, {}))
        if cap_config:
            min_probability = float(cap_config.get("min_probability", threshold) or threshold)
            cap_probability = float(cap_config.get("cap_probability", probability) or probability)
            if probability >= min_probability:
                updated["probability"] = min(probability, cap_probability)

        gated_subtypes = set(policy.get("require_regime_confirmation_for_subtypes", []))
        if subtype in gated_subtypes:
            high_vol = float(updated.get("reg_high_vol_state", 0.0) or 0.0) >= 1.0
            trending = float(updated.get("reg_trending_state", 0.0) or 0.0) >= 1.0
            if high_vol or not trending:
                continue

        filtered.append(updated)

    return filtered


def _time_bucket(row: dict[str, Any]) -> str:
    existing = row.get("time_bucket")
    if existing:
        return str(existing)
    value = row.get("entry_time")
    if not value:
        return "unknown"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    return parsed.strftime("%H:%M")
