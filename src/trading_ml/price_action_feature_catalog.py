from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from trading_ml.paths import CONFIGS_DIR

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


CATALOG_FILENAME = "price_action_feature_catalog.toml"

DEFAULT_GROUP_ORDER = [
    "volatility",
    "momentum",
    "candles",
    "structure",
    "liquidity",
    "auction",
]

KEYWORD_GROUP_MAP: dict[str, tuple[str, ...]] = {
    "reclaim": ("structure", "momentum", "liquidity"),
    "pivot": ("structure", "momentum"),
    "opening": ("auction", "structure"),
    "open": ("auction", "structure"),
    "break": ("momentum", "volatility"),
    "continuation": ("momentum", "candles"),
    "trend": ("auction", "momentum", "structure"),
    "vwap": ("structure", "liquidity"),
    "time": ("auction",),
    "volume": ("volatility", "momentum", "auction"),
    "wick": ("candles", "structure"),
    "chop": ("auction", "structure", "candles"),
    "repair": ("structure", "liquidity"),
    "followthrough": ("momentum", "candles"),
    "liquidity": ("liquidity",),
    "auction": ("auction",),
    "structure": ("structure",),
    "momentum": ("momentum",),
    "volatility": ("volatility",),
    "candle": ("candles",),
}

CLUSTER_GROUP_MAP: dict[str, tuple[str, ...]] = {
    "no_follow_through": ("momentum", "candles", "auction"),
    "weak_continuation": ("momentum", "candles", "auction"),
    "deep_retrace_failure": ("structure", "liquidity", "auction"),
    "no_reclaim_edge": ("liquidity", "structure", "momentum"),
    "failed_breakout": ("liquidity", "momentum", "structure"),
}

GROUP_LABS: dict[str, str] = {
    "volatility": "volatility_shape_lab",
    "momentum": "momentum_followthrough_lab",
    "candles": "candle_structure_lab",
    "structure": "structure_repair_lab",
    "liquidity": "liquidity_context_lab",
    "auction": "auction_context_lab",
}

RISK_RANK = {"low": 0, "medium": 1, "high": 2}


def catalog_path() -> Path:
    return CONFIGS_DIR / CATALOG_FILENAME


@lru_cache(maxsize=1)
def load_feature_catalog(path: str | Path | None = None) -> dict[str, Any]:
    catalog_file = Path(path) if path is not None else catalog_path()
    if not catalog_file.exists():
        return {"meta": {}, "groups": {}, "features": {}, "restricted": {}}
    with catalog_file.open("rb") as handle:
        raw = tomllib.load(handle)
    return _normalize_catalog(raw)


def build_feature_catalog_index(path: str | Path | None = None) -> dict[str, Any]:
    catalog = load_feature_catalog(path)
    index: dict[str, dict[str, Any]] = {}
    for name, spec in catalog.get("features", {}).items():
        index[name] = {
            **dict(spec),
            "feature_name": name,
            "group": spec.get("category", ""),
        }
    return index


def build_catalog_feature_proposals(
    strategy_notes: str,
    *,
    top_cluster: dict[str, Any] | None = None,
    bnr_spec: dict[str, Any] | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    intake = build_feature_catalog_intake(
        strategy_notes,
        bnr_spec=bnr_spec,
        top_cluster=top_cluster,
        limit=limit,
    )
    return {
        **intake,
        "feature_claim": _feature_claim(top_cluster or {}, intake.get("feature_catalog_candidates", [])),
    }


def build_strategy_intake(strategy_notes: str, bnr_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_feature_catalog_intake(strategy_notes, bnr_spec=bnr_spec)


def build_feature_catalog_intake(
    strategy_notes: str,
    *,
    bnr_spec: dict[str, Any] | None = None,
    top_cluster: dict[str, Any] | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    catalog = load_feature_catalog()
    notes = (strategy_notes or "").strip()
    lower = notes.lower()
    selected_groups = _select_groups(lower, top_cluster or {}, catalog)
    feature_index = build_feature_catalog_index()
    candidates = _rank_feature_candidates(feature_index, selected_groups, lower, top_cluster or {}, limit=limit)
    backlog = _feature_backlog(selected_groups, catalog)
    research_questions = _build_research_questions(lower, top_cluster or {}, candidates)
    return {
        "status": "complete" if notes else "seeded_from_default_bnr",
        "catalog_version": dict(catalog.get("meta", {}) or {}).get("version", 1),
        "strategy_notes": notes,
        "selected_feature_groups": selected_groups,
        "feature_backlog": backlog,
        "feature_catalog_candidates": candidates,
        "feature_catalog_groups": {
            group: dict(catalog.get("groups", {}).get(group, {}))
            for group in selected_groups
        },
        "research_questions": research_questions,
        "next_feature_labs": _next_feature_labs(selected_groups),
        "restricted_feature_areas": dict(catalog.get("restricted", {}) or {}),
        "bnr_setup_name": (bnr_spec or {}).get("setup", {}).get("name", "BNR"),
    }


def _normalize_catalog(raw: dict[str, Any]) -> dict[str, Any]:
    groups = {
        name: {
            **dict(spec),
            "features": list(dict(spec).get("features", []) or []),
        }
        for name, spec in dict(raw.get("groups", {}) or {}).items()
    }
    features = {
        name: {
            **dict(spec),
            "category": str(dict(spec).get("category", "")),
            "families": list(dict(spec).get("families", []) or []),
            "inputs": list(dict(spec).get("inputs", []) or []),
            "derived_from": list(dict(spec).get("derived_from", []) or []),
            "invalidates": list(dict(spec).get("invalidates", []) or []),
            "related_clusters": list(dict(spec).get("related_clusters", []) or []),
        }
        for name, spec in dict(raw.get("features", {}) or {}).items()
    }
    return {
        "meta": dict(raw.get("meta", {}) or {}),
        "groups": groups,
        "features": features,
        "restricted": dict(raw.get("restricted", {}) or {}),
    }


def _select_groups(lower_notes: str, top_cluster: dict[str, Any], catalog: dict[str, Any]) -> list[str]:
    selected: list[str] = []
    for keyword, groups in KEYWORD_GROUP_MAP.items():
        if keyword in lower_notes:
            for group in groups:
                if group in catalog.get("groups", {}) and group not in selected:
                    selected.append(group)
    cluster_family = str(top_cluster.get("family", "") or "")
    recommended_family = str(top_cluster.get("recommended_family", "") or "")
    for family_key in (cluster_family, recommended_family):
        for group in CLUSTER_GROUP_MAP.get(family_key, ()):
            if group in catalog.get("groups", {}) and group not in selected:
                selected.append(group)
    if not selected:
        for group in DEFAULT_GROUP_ORDER:
            if group in catalog.get("groups", {}):
                selected.append(group)
    if "auction" not in selected and any(token in lower_notes for token in ("open", "opening", "session")):
        if "auction" in catalog.get("groups", {}):
            selected.append("auction")
    return selected


def _rank_feature_candidates(
    feature_index: dict[str, dict[str, Any]],
    selected_groups: list[str],
    lower_notes: str,
    top_cluster: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    cluster_focus = " ".join(
        str(part)
        for part in (
            top_cluster.get("family", ""),
            top_cluster.get("recommended_family", ""),
            " ".join(top_cluster.get("recommended_focus", []) or []),
            top_cluster.get("dominant_subtype", ""),
            top_cluster.get("dominant_time_bucket", ""),
        )
        if part
    ).lower()
    requested_terms = set(lower_notes.split()) | set(cluster_focus.split())
    ranked: list[dict[str, Any]] = []
    for name, spec in feature_index.items():
        group = str(spec.get("category", ""))
        if group not in selected_groups:
            continue
        score = _feature_score(name, spec, requested_terms, cluster_focus, selected_groups)
        ranked.append(
            {
                "feature_name": name,
                "group": group,
                "category": spec.get("category", group),
                "concept": spec.get("concept", ""),
                "families": list(spec.get("families", []) or []),
                "risk": spec.get("risk", "medium"),
                "curve_fit_risk": spec.get("curve_fit_risk", "medium"),
                "support_requirement": spec.get("support_requirement", "medium"),
                "description": spec.get("description", ""),
                "inputs": list(spec.get("inputs", []) or []),
                "related_clusters": list(spec.get("related_clusters", []) or []),
                "score": score,
            }
        )
    ranked.sort(
        key=lambda row: (
            -int(row["score"]),
            RISK_RANK.get(str(row.get("risk", "medium")), 1),
            RISK_RANK.get(str(row.get("support_requirement", "medium")), 1),
            str(row["feature_name"]),
        )
    )
    return ranked[:limit]


def _feature_score(
    name: str,
    spec: dict[str, Any],
    requested_terms: set[str],
    cluster_focus: str,
    selected_groups: list[str],
) -> int:
    haystack = " ".join(
        [
            name,
            str(spec.get("category", "")),
            str(spec.get("concept", "")),
            str(spec.get("description", "")),
            " ".join(spec.get("families", []) or []),
            " ".join(spec.get("related_clusters", []) or []),
        ]
    ).lower()
    score = 0
    for term in requested_terms:
        if len(term) < 4:
            continue
        if term in haystack:
            score += 2
    if str(spec.get("category", "")) in selected_groups:
        score += 3
    if cluster_focus and any(token in haystack for token in cluster_focus.split()):
        score += 2
    if str(spec.get("risk", "medium")) == "low":
        score += 1
    if str(spec.get("support_requirement", "medium")) == "low":
        score += 1
    return score


def _feature_backlog(selected_groups: list[str], catalog: dict[str, Any]) -> dict[str, list[str]]:
    groups = dict(catalog.get("groups", {}) or {})
    return {
        group: list(dict(groups.get(group, {}) or {}).get("features", []) or [])
        for group in selected_groups
        if group in groups
    }


def _build_research_questions(lower_notes: str, top_cluster: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    family = str(top_cluster.get("family", "unknown") or "unknown")
    recommended = str(top_cluster.get("recommended_family", "") or "")
    questions = [
        f"Which catalog features separate {family} attempts from the profitable BNR cases?",
        f"Which feature families best explain the cluster's recommended follow-up: {recommended or 'feature'}?",
        "Which points-in-time signals improve BNR eligibility without adding path-specific leakage?",
    ]
    if "vwap" in lower_notes:
        questions.append("How does VWAP location interact with reclaim quality and continuation quality?")
    if "volume" in lower_notes:
        questions.append("Does volume expansion at the break or reclaim survive CPCV and plumbing controls?")
    if candidates:
        questions.append(f"Can {candidates[0]['feature_name']} improve the cluster's weakest failure mode?")
    return questions


def _next_feature_labs(selected_groups: list[str]) -> list[str]:
    return [GROUP_LABS[group] for group in selected_groups if group in GROUP_LABS]


def _feature_claim(top_cluster: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "No catalog-backed feature candidates available yet."
    top_family = str(top_cluster.get("family", "BNR") or "BNR")
    lead = candidates[0]
    return (
        f"Target {top_family} failures with {lead['feature_name']} "
        f"from the {lead['group']} catalog family."
    )
