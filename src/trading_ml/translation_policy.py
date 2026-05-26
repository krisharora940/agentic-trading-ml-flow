from __future__ import annotations

from typing import Any


def get_sizing_policies() -> list[dict[str, Any]]:
    return [
        {"name": "binary_threshold_v1", "kind": "binary"},
        {
            "name": "confidence_linear_v1",
            "kind": "confidence_linear",
            "min_size": 0.50,
            "max_size": 1.50,
        },
        {
            "name": "confidence_tiered_v1",
            "kind": "confidence_tiered",
            "low_size": 0.50,
            "mid_size": 1.00,
            "high_size": 1.50,
            "high_threshold_offset": 0.20,
        },
        {
            "name": "fractional_kelly_proxy_v1",
            "kind": "fractional_kelly_proxy",
            "min_size": 0.25,
            "max_size": 1.25,
            "kelly_fraction": 0.50,
        },
    ]


def get_regime_throttle_policies() -> list[dict[str, Any]]:
    return [
        {"name": "none"},
        {"name": "suppress_high_vol_v1", "max_high_vol_state": 0.0},
        {"name": "trending_only_v1", "require_trending_state": 1.0},
        {
            "name": "high_vol_or_non_trending_off_v1",
            "max_high_vol_state": 0.0,
            "require_trending_state": 1.0,
        },
    ]


def get_regime_size_policies() -> list[dict[str, Any]]:
    return [
        {"name": "none"},
        {
            "name": "trend_vol_scale_v1",
            "good_multiplier": 1.00,
            "ambiguous_multiplier": 0.75,
            "bad_multiplier": 0.35,
        },
        {
            "name": "high_vol_haircut_v1",
            "good_multiplier": 1.00,
            "ambiguous_multiplier": 0.80,
            "bad_multiplier": 0.50,
        },
    ]


def get_sizing_policy(name: str | None) -> dict[str, Any]:
    target = name or "binary_threshold_v1"
    for policy in get_sizing_policies():
        if policy["name"] == target:
            return policy
    raise KeyError(f"Unknown sizing policy: {target}")


def get_regime_throttle_policy(name: str | None) -> dict[str, Any]:
    target = name or "none"
    for policy in get_regime_throttle_policies():
        if policy["name"] == target:
            return policy
    raise KeyError(f"Unknown regime throttle policy: {target}")


def get_regime_size_policy(name: str | None) -> dict[str, Any]:
    target = name or "none"
    for policy in get_regime_size_policies():
        if policy["name"] == target:
            return policy
    raise KeyError(f"Unknown regime size policy: {target}")


def compute_position_size(
    probability: float, *, threshold: float, policy_name: str | None
) -> float:
    policy = get_sizing_policy(policy_name)
    if probability < threshold:
        return 0.0
    kind = str(policy.get("kind", "binary"))
    if kind == "binary":
        return 1.0
    if kind == "confidence_linear":
        min_size = float(policy.get("min_size", 0.5) or 0.5)
        max_size = float(policy.get("max_size", 1.5) or 1.5)
        if threshold >= 1.0:
            return min_size
        progress = max(
            0.0, min(1.0, (probability - threshold) / max(1.0 - threshold, 1e-9))
        )
        return min_size + (max_size - min_size) * progress
    if kind == "confidence_tiered":
        high_offset = float(policy.get("high_threshold_offset", 0.2) or 0.2)
        if probability >= min(1.0, threshold + high_offset):
            return float(policy.get("high_size", 1.5) or 1.5)
        if probability >= min(1.0, threshold + high_offset / 2.0):
            return float(policy.get("mid_size", 1.0) or 1.0)
        return float(policy.get("low_size", 0.5) or 0.5)
    if kind == "fractional_kelly_proxy":
        min_size = float(policy.get("min_size", 0.25) or 0.25)
        max_size = float(policy.get("max_size", 1.5) or 1.5)
        kelly_fraction = float(policy.get("kelly_fraction", 0.5) or 0.5)
        if threshold >= 1.0:
            return min_size
        normalized_edge = max(
            0.0, min(1.0, (probability - threshold) / max(1.0 - threshold, 1e-9))
        )
        return max(min_size, min(max_size, max_size * kelly_fraction * normalized_edge))
    return 1.0


def allow_signal_for_regime(record: dict[str, Any], *, policy_name: str | None) -> bool:
    policy = get_regime_throttle_policy(policy_name)
    if "max_high_vol_state" in policy and float(
        record.get("reg_high_vol_state", 0.0) or 0.0
    ) > float(policy["max_high_vol_state"]):
        return False
    if "require_trending_state" in policy and float(
        record.get("reg_trending_state", 0.0) or 0.0
    ) < float(policy["require_trending_state"]):
        return False
    return True


def compute_regime_size_multiplier(
    record: dict[str, Any], *, policy_name: str | None
) -> float:
    policy = get_regime_size_policy(policy_name)
    if policy["name"] == "none":
        return 1.0
    high_vol = float(record.get("reg_high_vol_state", 0.0) or 0.0) >= 1.0
    trending = float(record.get("reg_trending_state", 0.0) or 0.0) >= 1.0
    if trending and not high_vol:
        return float(policy.get("good_multiplier", 1.0) or 1.0)
    if trending or not high_vol:
        return float(policy.get("ambiguous_multiplier", 0.75) or 0.75)
    return float(policy.get("bad_multiplier", 0.5) or 0.5)
