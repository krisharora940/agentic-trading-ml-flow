from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trading_ml.config import load_bnr_config
from trading_ml.bnr_subtypes import (
    classify_candidate_subtype,
    filter_candidates_by_subtype,
)
from trading_ml.feature_validation import build_feature_validation
from trading_ml.feature_families import apply_feature_family
from trading_ml.market_structure_lab import build_market_structure_lab
from trading_ml.model_diagnostics_lab import build_model_diagnostics_lab
from trading_ml.stage2_bnr import calculate_bnr_zones, generate_breakout_candidates
from trading_ml.stage2_data import (
    build_data_quality_report,
    load_ohlcv_file,
    regular_session,
)
from trading_ml.stage2_features import build_feature_matrix
from trading_ml.stage2_labeling import label_candidates
from trading_ml.stage2_modeling import train_baseline_classifier


@dataclass(slots=True)
class Stage2Config:
    source_path: str
    symbol: str = "MNQ"
    timeframe: str = "30s"
    timezone: str = "America/New_York"
    earliest_trigger_time: str = "09:32:00"
    latest_trigger_time: str = "11:00:00"
    horizon_bars: int = 20
    stop_multiple: float = 1.0
    target_multiple: float = 1.5
    break_buffer_points: float = 0.0
    spec_name: str = "BNR"
    model_family: str = "linear_baseline"
    feature_family: str = "all_features"
    setup_subtype: str = "all_subtypes"

    @classmethod
    def from_bnr_config(cls, source_path: str, **overrides: Any) -> "Stage2Config":
        config = load_bnr_config()
        label_v1 = config["label_v1"]
        phases_entry = config["phases"]["entry"]
        setup = config["setup"]
        payload: dict[str, Any] = {
            "source_path": source_path,
            "symbol": setup["symbol"],
            "timeframe": setup["decision_unit"].split("_", 1)[0],
            "timezone": setup["timezone"],
            "earliest_trigger_time": phases_entry["earliest_entry_time"],
            "latest_trigger_time": "11:00:00",
            "horizon_bars": label_v1["horizon_bars"],
            "stop_multiple": label_v1["stop_r"],
            "target_multiple": label_v1["target_r"],
            "break_buffer_points": 0.0,
            "spec_name": setup["name"],
        }
        payload.update(overrides)
        return cls(**payload)


def run_stage2_research_engine(config: Stage2Config) -> dict[str, Any]:
    pd = _require_pandas()
    bars = load_ohlcv_file(
        config.source_path,
        symbol=config.symbol,
        timeframe=config.timeframe,
        timezone=config.timezone,
    )
    report = build_data_quality_report(
        bars,
        source_path=config.source_path,
        symbol=config.symbol,
        timeframe=config.timeframe,
        timezone=config.timezone,
    )
    rth = regular_session(bars)
    zones = calculate_bnr_zones(rth, symbol=config.symbol, timeframe=config.timeframe)
    candidates = generate_breakout_candidates(
        rth,
        zones,
        timeframe=config.timeframe,
        earliest_trigger_time=config.earliest_trigger_time,
        latest_trigger_time=config.latest_trigger_time,
        break_buffer_points=config.break_buffer_points,
    )
    candidates = filter_candidates_by_subtype(candidates, config.setup_subtype)
    labels = label_candidates(
        rth,
        candidates,
        horizon_bars=config.horizon_bars,
        stop_multiple=config.stop_multiple,
        target_multiple=config.target_multiple,
    )
    features, feature_audits = build_feature_matrix(rth, candidates)
    if not features.empty:
        subtype_map = {
            candidate.candidate_id: classify_candidate_subtype(candidate)
            for candidate in candidates
        }
        features["setup_subtype"] = features["candidate_id"].map(subtype_map)
    features = apply_feature_family(features, config.feature_family)
    labels_df = pd.DataFrame([label.to_dict() for label in labels])
    model_summary = (
        train_baseline_classifier(features, labels_df, model_family=config.model_family)
        if not labels_df.empty
        else None
    )
    feature_validation = build_feature_validation(
        features.to_dict(orient="records"), labels_df.to_dict(orient="records")
    )
    prediction_records = model_summary.prediction_records if model_summary else []
    market_structure_lab = build_market_structure_lab(
        [candidate.to_dict() for candidate in candidates],
        labels_df.to_dict(orient="records"),
    )
    model_diagnostics = build_model_diagnostics_lab(
        prediction_records,
        features.to_dict(orient="records"),
        labels_df.to_dict(orient="records"),
        model_family=config.model_family,
    )
    return {
        "config": asdict(config),
        "bnr_spec": load_bnr_config(),
        "data_quality": report.to_dict(),
        "zone_count": len(zones),
        "candidate_count": len(candidates),
        "label_summary": _label_summary(labels_df),
        "feature_audit": _feature_audit_summary(feature_audits),
        "feature_validation": feature_validation,
        "market_structure_lab": market_structure_lab,
        "model_summary": (
            model_summary.to_dict() if model_summary else {"status": "no_labels"}
        ),
        "model_diagnostics": model_diagnostics,
        "features_records": features.to_dict(orient="records"),
        "labels_records": labels_df.to_dict(orient="records"),
        "sample_zones": [zone.to_dict() for zone in zones[:3]],
        "sample_candidates": [candidate.to_dict() for candidate in candidates[:3]],
        "subtype_counts": _subtype_counts(candidates),
        "labels_preview": labels_df.head(5).to_dict(orient="records"),
    }


def write_stage2_report(result: dict[str, Any], output_path: str | Path) -> None:
    import json

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True))


def _label_summary(labels_df: Any) -> dict[str, Any]:
    if labels_df.empty:
        return {"rows": 0}
    return {
        "rows": int(len(labels_df)),
        "positive_rate": float(labels_df["label"].mean()),
        "outcomes": {
            str(key): int(value)
            for key, value in labels_df["outcome"].value_counts().to_dict().items()
        },
        "avg_pnl_r": float(labels_df["pnl_r"].mean()),
    }


def _feature_audit_summary(audits: list[Any]) -> dict[str, Any]:
    failed = [audit for audit in audits if audit.status != "pass"]
    return {
        "rows": len(audits),
        "failed": len(failed),
        "issues": sorted({issue for audit in failed for issue in audit.issues}),
    }


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Stage 2 requires pandas.") from exc
    return pd


def _subtype_counts(candidates: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        subtype = classify_candidate_subtype(candidate)
        counts[subtype] = counts.get(subtype, 0) + 1
    return counts
