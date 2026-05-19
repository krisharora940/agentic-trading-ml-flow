from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.stage2_modeling import score_model_split
from trading_ml.validation_splits import build_walk_forward_splits


def bucket_monotonicity(prediction_rows: list[dict], bucket_count: int = 5) -> dict:
    try:
        import pandas as pd
    except ImportError:
        return {"status": "pending"}
    frame = pd.DataFrame(prediction_rows)
    if frame.empty or len(frame) < bucket_count:
        return {"status": "pending"}
    ranked = frame.sort_values("probability").reset_index(drop=True)
    ranked["bucket"] = pd.qcut(ranked.index, q=min(bucket_count, len(ranked)), duplicates="drop")
    bucket_rows = []
    for _, group in ranked.groupby("bucket", observed=False):
        bucket_rows.append(
            {
                "avg_probability": float(group["probability"].mean()),
                "hit_rate": float(group["label"].mean()),
                "avg_pnl_r": float(group["pnl_r"].mean()),
                "count": int(len(group)),
            }
        )
    monotonic = all(
        bucket_rows[idx]["hit_rate"] <= bucket_rows[idx + 1]["hit_rate"]
        for idx in range(len(bucket_rows) - 1)
    )
    return {"status": "complete", "monotonic_hit_rate": monotonic, "buckets": bucket_rows}


def regime_conditional_summary(prediction_rows: list[dict]) -> dict:
    try:
        import pandas as pd
    except ImportError:
        return {"status": "pending"}
    frame = pd.DataFrame(prediction_rows)
    if frame.empty or "prior_close_gap" not in frame:
        return {"status": "pending"}
    regimes = {
        "gap_up": frame[frame["prior_close_gap"] > 0],
        "gap_down": frame[frame["prior_close_gap"] < 0],
    }
    rows = {}
    for name, group in regimes.items():
        if len(group) < 5 or group["label"].nunique() < 2:
            continue
        corr = group["probability"].corr(group["label"])
        rows[name] = {
            "count": int(len(group)),
            "conditional_ic": float(corr) if corr == corr else None,
            "avg_pnl_r": float(group["pnl_r"].mean()),
        }
    return {"status": "complete" if rows else "pending", "regimes": rows}


def main() -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("cross-model analysis requires pandas") from exc

    state = build_agent_loop_state()
    config = dict(state["stage2_config"])
    config["target_multiple"] = 1.5
    result = run_stage2_research_engine(Stage2Config(**config))
    features = pd.DataFrame(result["features_records"])
    labels = pd.DataFrame(result["labels_records"])
    merged = features.merge(labels, on="candidate_id", how="inner")
    feature_cols = [
        col
        for col in merged.columns
        if col not in {"candidate_id", "session_date", "label", "outcome", "entry_time", "exit_time", "entry_price", "stop_price", "target_price", "exit_price", "bars_held", "mfe", "mae", "pnl_r"}
        and pd.api.types.is_numeric_dtype(merged[col])
        and not merged[col].isna().all()
    ]
    folds_input, metadata = build_walk_forward_splits(merged)
    active_families = [
        {"family": "linear_baseline", "status": "active"},
        {"family": "gbm", "status": "active"},
        {"family": "dl", "status": "not_implemented"},
        {"family": "causal", "status": "not_implemented"},
    ]
    family_results = []
    for family in active_families:
        if family["status"] != "active":
            family_results.append({"model_family": family["family"], "status": family["status"]})
            continue
        fold_rows = []
        stitched_rows = []
        for train, test, fold_meta in folds_input[:2]:
            if train.empty or test.empty or train["label"].nunique() < 2 or test["label"].nunique() < 2:
                continue
            scored = score_model_split(train, test, model_family=family["family"], feature_cols=feature_cols)
            pred = scored["prediction_frame"].copy()
            pred["fold"] = fold_meta.fold
            stitched_rows.extend(pred.to_dict(orient="records"))
            fold_rows.append(
                {
                    "fold": fold_meta.fold,
                    "train_rows": fold_meta.train_rows,
                    "test_rows": fold_meta.test_rows,
                    **scored["metrics"],
                }
            )
        roc_values = [row.get("roc_auc") for row in fold_rows if row.get("roc_auc") is not None]
        family_results.append(
            {
                "model_family": family["family"],
                "status": "evaluated" if fold_rows else "pending",
                "folds": fold_rows,
                "fold_stability": {
                    "roc_auc_mean": mean(roc_values) if roc_values else None,
                    "roc_auc_std": pstdev(roc_values) if len(roc_values) > 1 else 0.0 if roc_values else None,
                },
                "bucket_monotonicity": bucket_monotonicity(stitched_rows),
                "regime_conditional": regime_conditional_summary(stitched_rows),
                "advancement_recommendation": "advance_to_signal_stage" if roc_values and mean(roc_values) >= 0.55 else "freeze",
            }
        )

    payload = {
        "source": "cross_model_analysis_30s_mnq",
        "label_target_multiple": 1.5,
        "fold_backend": metadata.get("backend"),
        "folds_evaluated": 2,
        "results": family_results,
    }
    output = Path("reports/cross_model_analysis.json")
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
