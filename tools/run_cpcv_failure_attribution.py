from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trading_ml.paths import REPORTS_DIR


SOURCE_REPORT = REPORTS_DIR / "exploration_benchmark_diagnostics.json"
OUTPUT_REPORT = REPORTS_DIR / "cpcv_failure_attribution.json"
MAX_PATHS = 6


def main() -> None:
    payload = build_cpcv_failure_attribution(SOURCE_REPORT)
    OUTPUT_REPORT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


def build_cpcv_failure_attribution(source_report: Path) -> dict[str, Any]:
    if not source_report.exists():
        return _incomplete(
            reason="missing_source_diagnostics",
            source_report=source_report,
            next_step="run exploration benchmark diagnostics with CPCV persistence enabled",
        )

    diagnostics = json.loads(source_report.read_text(encoding="utf-8"))
    cpcv = dict(diagnostics.get("cpcv", {}) or {})
    worst_paths = list(cpcv.get("worst_paths", []) or [])
    best_paths = list(cpcv.get("best_paths", []) or [])
    candidate_paths = worst_paths[:3] + best_paths[:3]
    artifact_refs = [row.get("rows_artifact") for row in candidate_paths if row.get("rows_artifact")]
    if not artifact_refs:
        return _incomplete(
            reason="missing_persisted_cpcv_rows",
            source_report=source_report,
            next_step="rerun main validation with CPCV row persistence enabled",
            cpcv=cpcv,
        )

    loaded_paths: list[dict[str, Any]] = []
    missing_refs: list[str] = []
    for row in candidate_paths[:MAX_PATHS]:
        ref = row.get("rows_artifact")
        if not ref:
            continue
        path = Path(ref)
        if not path.exists():
            missing_refs.append(ref)
            continue
        loaded_paths.append({"summary": row, "rows": json.loads(path.read_text(encoding="utf-8"))})

    if not loaded_paths:
        return _incomplete(
            reason="persisted_cpcv_rows_unavailable",
            source_report=source_report,
            next_step="rerun main validation with CPCV row persistence enabled",
            cpcv=cpcv,
            missing_artifacts=missing_refs,
        )

    worst_loaded = loaded_paths[0]
    dominant_axes = _dominant_failure_axes(worst_loaded["rows"])
    largest_loss_cluster = worst_loaded["summary"].get("largest_loss_cluster_r")
    payload = {
        "status": "complete",
        "source_diagnostics": str(source_report),
        "cpcv_status": cpcv.get("status"),
        "failure_summary": {
            "pbo": cpcv.get("pbo"),
            "mean_path_pnl_r": cpcv.get("mean_total_pnl_r"),
            "median_path_pnl_r": cpcv.get("median_total_pnl_r"),
            "min_path_pnl_r": cpcv.get("min_path_pnl_r"),
            "positive_path_rate": cpcv.get("path_positive_rate"),
            "failure_type": _failure_type(cpcv),
        },
        "path_distribution": cpcv.get("distribution", {}),
        "worst_paths": worst_paths,
        "best_paths": best_paths,
        "dominant_failure_axes": dominant_axes,
        "largest_loss_cluster": largest_loss_cluster,
        "recommended_next_family": "diagnose_tail_regime_or_subtype",
        "blocked_actions": [
            "promotion",
            "holdout_confirmation",
            "threshold_optimization_only",
            "sizing_optimization_only",
        ],
    }
    if missing_refs:
        payload["missing_artifacts"] = missing_refs
    return payload


def _incomplete(
    *,
    reason: str,
    source_report: Path,
    next_step: str,
    cpcv: dict[str, Any] | None = None,
    missing_artifacts: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "status": "insufficient_artifacts",
        "reason": reason,
        "source_diagnostics": str(source_report),
        "required_next_step": next_step,
    }
    if cpcv is not None:
        payload["cpcv_status"] = cpcv.get("status")
        payload["failure_summary"] = {
            "pbo": cpcv.get("pbo"),
            "mean_path_pnl_r": cpcv.get("mean_total_pnl_r"),
            "median_path_pnl_r": cpcv.get("median_total_pnl_r"),
            "min_path_pnl_r": cpcv.get("min_path_pnl_r"),
            "positive_path_rate": cpcv.get("path_positive_rate"),
            "failure_type": _failure_type(cpcv),
        }
        payload["path_distribution"] = cpcv.get("distribution", {})
        payload["worst_paths"] = cpcv.get("worst_paths", [])
        payload["best_paths"] = cpcv.get("best_paths", [])
    if missing_artifacts:
        payload["missing_artifacts"] = missing_artifacts
    return payload


def _failure_type(cpcv: dict[str, Any]) -> str:
    mean_pnl = float(cpcv.get("mean_total_pnl_r", 0.0) or 0.0)
    median_pnl = float(cpcv.get("median_total_pnl_r", 0.0) or 0.0)
    min_pnl = float(cpcv.get("min_path_pnl_r", 0.0) or 0.0)
    positive_rate = float(cpcv.get("path_positive_rate", 0.0) or 0.0)
    if median_pnl > 0 and positive_rate >= 0.60 and mean_pnl <= 0 and min_pnl < -5.0:
        return "tail_path_fragility"
    if positive_rate < 0.50:
        return "weak_edge"
    return "mixed_path_instability"


def _dominant_failure_axes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    losers = [row for row in rows if float(row.get("executed_pnl_r", 0.0) or 0.0) < 0]
    frame = losers or rows
    return {
        "subtype": _top_group(frame, "setup_subtype"),
        "time_of_day": _top_group(frame, "time_bucket"),
        "volatility_regime": _binary_state_group(frame, "reg_high_vol_state"),
        "trend_regime": _binary_state_group(frame, "reg_trending_state"),
        "threshold_bucket": _probability_bucket(frame),
        "probability_bucket": _probability_bucket(frame),
    }


def _top_group(rows: list[dict[str, Any]], column: str) -> dict[str, Any] | None:
    counts: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(column, "unknown"))
        bucket = counts.setdefault(key, {"key": key, "trade_count": 0, "total_pnl_r": 0.0})
        bucket["trade_count"] += 1
        bucket["total_pnl_r"] += float(row.get("executed_pnl_r", 0.0) or 0.0)
    if not counts:
        return None
    return max(counts.values(), key=lambda item: (item["trade_count"], -item["total_pnl_r"]))


def _binary_state_group(rows: list[dict[str, Any]], column: str) -> dict[str, Any] | None:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        copy[column] = "on" if bool(row.get(column)) else "off"
        normalized.append(copy)
    return _top_group(normalized, column)


def _probability_bucket(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"trade_count": 0, "total_pnl_r": 0.0})
    for row in rows:
        prob = float(row.get("probability", 0.0) or 0.0)
        if prob < 0.55:
            key = "[0.45,0.55)"
        elif prob < 0.65:
            key = "[0.55,0.65)"
        else:
            key = "[0.65,1.00]"
        buckets[key]["trade_count"] += 1
        buckets[key]["total_pnl_r"] += float(row.get("executed_pnl_r", 0.0) or 0.0)
    if not buckets:
        return None
    key, value = max(buckets.items(), key=lambda item: (item[1]["trade_count"], -item[1]["total_pnl_r"]))
    return {"key": key, **value}


if __name__ == "__main__":
    main()
