from __future__ import annotations

import random
from typing import Any

from trading_ml.config import load_bnr_config
from trading_ml.cpcv_analysis import build_cpcv_audit
from trading_ml.config import load_global_config
from trading_ml.deflated_sharpe_analysis import build_deflated_sharpe_audit
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.utility_analysis import compute_execution_utility
from trading_ml.validation_splits import build_walk_forward_splits


def build_validation_audit(
    stage2_result: dict[str, Any],
    search_results: dict[str, Any],
    controller_state: dict[str, Any] | None = None,
    artifact_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    features = list(stage2_result.get("features_records", []))
    labels = list(stage2_result.get("labels_records", []))
    walk_forward = _walk_forward_check(features, labels)
    cpcv = build_cpcv_audit(stage2_result, artifact_context=artifact_context)
    deflated_sharpe = build_deflated_sharpe_audit(
        stage2_result,
        walk_forward,
        cpcv,
        search_results,
        controller_state=controller_state,
    )
    purging = _purging_check(labels)
    multiple_testing = _multiple_testing_check(search_results)
    random_signal_plumbing = _random_signal_plumbing_check(
        walk_forward, search_results, controller_state=controller_state
    )
    overfitting = _overfitting_check(
        walk_forward, cpcv, multiple_testing, deflated_sharpe
    )
    return {
        "walk_forward": walk_forward,
        "cpcv": cpcv,
        "deflated_sharpe": deflated_sharpe,
        "purging": purging,
        "multiple_testing": multiple_testing,
        "random_signal_plumbing": random_signal_plumbing,
        "overfitting": overfitting,
    }


def _walk_forward_check(
    features: list[dict[str, Any]], labels: list[dict[str, Any]]
) -> dict[str, Any]:
    try:
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    if not features or not labels:
        return {"status": "pending", "reason": "missing_artifacts"}

    features_df = pd.DataFrame(features)
    labels_df = pd.DataFrame(labels)
    merged = features_df.merge(labels_df, on="candidate_id", how="inner")
    if merged.empty:
        return {"status": "pending", "reason": "empty_merged_dataset"}

    folds_input, metadata = build_walk_forward_splits(merged)
    if not folds_input:
        return {"status": "pending", "reason": "insufficient_sessions", **metadata}

    folds: list[dict[str, Any]] = []
    stitched_predictions: list[dict[str, Any]] = []
    feature_cols = [
        col
        for col in merged.columns
        if col
        not in {
            "candidate_id",
            "session_date",
            "label",
            "outcome",
            "entry_time",
            "exit_time",
            "entry_price",
            "stop_price",
            "target_price",
            "exit_price",
            "bars_held",
            "mfe",
            "mae",
            "pnl_r",
        }
        and pd.api.types.is_numeric_dtype(merged[col])
        and not merged[col].isna().all()
    ]
    for train, test, fold_meta in folds_input:
        if (
            train.empty
            or test.empty
            or train["label"].nunique() < 2
            or test["label"].nunique() < 2
        ):
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        model.fit(train[feature_cols], train["label"])
        probabilities = model.predict_proba(test[feature_cols])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        prediction_frame = test[
            [
                "candidate_id",
                "session_date",
                "direction",
                "setup_subtype",
                "label",
                "pnl_r",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "stop_price",
                "target_price",
                "bars_held",
            ]
        ].copy()
        prediction_frame["probability"] = probabilities
        prediction_frame["prediction"] = predictions
        prediction_frame["fold"] = fold_meta.fold
        stitched_predictions.extend(prediction_frame.to_dict(orient="records"))
        folds.append(
            {
                **fold_meta.to_dict(),
                "roc_auc": float(roc_auc_score(test["label"], probabilities)),
                "precision": float(
                    precision_score(test["label"], predictions, zero_division=0)
                ),
            }
        )

    if len(folds) < 2:
        return {
            "status": "pending",
            "reason": "insufficient_valid_folds",
            "fold_count": len(folds),
            **metadata,
        }

    mean_roc_auc = sum(item["roc_auc"] for item in folds) / len(folds)
    status = "pass" if mean_roc_auc >= 0.55 else "fail"
    return {
        "status": status,
        "fold_count": len(folds),
        "mean_roc_auc": mean_roc_auc,
        **metadata,
        "folds": folds,
        "stitched_prediction_records": stitched_predictions,
    }


def _purging_check(labels: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    if not labels:
        return {"status": "pending", "reason": "missing_labels"}

    labels_df = pd.DataFrame(labels)
    if labels_df.empty or "entry_time" not in labels_df or "exit_time" not in labels_df:
        return {"status": "pending", "reason": "missing_time_bounds"}

    global_config = load_global_config()
    purging_bars = int(global_config["validation"].get("purging_bars", 0))
    labels_df["entry_time"] = pd.to_datetime(
        labels_df["entry_time"], errors="coerce", utc=True
    )
    labels_df["exit_time"] = pd.to_datetime(
        labels_df["exit_time"], errors="coerce", utc=True
    )
    labels_df = (
        labels_df.dropna(subset=["entry_time", "exit_time"])
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
    if len(labels_df) < 2:
        return {"status": "pass", "overlap_ratio": 0.0, "overlapping_pairs": 0}

    overlapping_pairs = 0
    for idx in range(1, len(labels_df)):
        if labels_df.iloc[idx]["entry_time"] < labels_df.iloc[idx - 1]["exit_time"]:
            overlapping_pairs += 1
    overlap_ratio = overlapping_pairs / max(len(labels_df) - 1, 1)
    if overlapping_pairs == 0:
        return {
            "status": "pass",
            "overlap_ratio": overlap_ratio,
            "overlapping_pairs": overlapping_pairs,
        }
    if purging_bars > 0:
        return {
            "status": "pass",
            "overlap_ratio": overlap_ratio,
            "overlapping_pairs": overlapping_pairs,
            "purging_bars": purging_bars,
        }
    return {
        "status": "fail",
        "overlap_ratio": overlap_ratio,
        "overlapping_pairs": overlapping_pairs,
        "purging_bars": purging_bars,
    }


def _multiple_testing_check(search_results: dict[str, Any]) -> dict[str, Any]:
    trial_count = int(search_results.get("trial_count", 0) or 0)
    accepted = dict(search_results.get("accepted_trial", {}) or {})
    if trial_count <= 0:
        return {"status": "pending", "reason": "no_trials"}
    if not accepted:
        return {
            "status": "fail",
            "trial_count": trial_count,
            "reason": "no_trial_cleared_controller",
        }

    ranked_trials = list(search_results.get("ranked_trials", []) or [])
    net_delta = float(accepted.get("net_delta_vs_baseline", 0.0) or 0.0)
    roc_delta = float(accepted.get("roc_auc_delta_vs_baseline", 0.0) or 0.0)
    trial_deltas = [
        float(row.get("net_delta_vs_baseline", 0.0) or 0.0)
        for row in ranked_trials
        if row.get("trial_id") != "baseline"
    ]
    centered = _centered_trial_deltas(trial_deltas)
    familywise_pvalue = _familywise_max_pvalue(centered, observed_best=net_delta)
    empirical_effect_floor = _empirical_effect_floor(trial_deltas)
    effect_above_floor = net_delta > empirical_effect_floor
    promotable_method = (
        net_delta > 0
        and roc_delta >= 0
        and effect_above_floor
        and familywise_pvalue <= 0.05
    )
    status = "pass" if promotable_method else "fail"
    return {
        "status": status,
        "method": "familywise_bootstrap_max_delta",
        "promotable_method": promotable_method,
        "trial_count": trial_count,
        "accepted_trial_id": accepted.get("trial_id"),
        "net_delta_vs_baseline": net_delta,
        "roc_auc_delta_vs_baseline": roc_delta,
        "familywise_pvalue": familywise_pvalue,
        "empirical_effect_floor": empirical_effect_floor,
        "effect_above_floor": effect_above_floor,
        "observed_trial_deltas": trial_deltas,
    }


def _random_signal_plumbing_check(
    walk_forward: dict[str, Any],
    search_results: dict[str, Any],
    *,
    controller_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = list(walk_forward.get("stitched_prediction_records", []) or [])
    if not records:
        return {"status": "pending", "reason": "missing_stitched_predictions"}

    try:
        import pandas as pd
    except ImportError:
        return {"status": "pending", "reason": "missing_dependencies"}

    frame = pd.DataFrame(records)
    required = {
        "candidate_id",
        "direction",
        "probability",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "stop_price",
        "pnl_r",
        "session_date",
    }
    if frame.empty or not required.issubset(frame.columns):
        return {"status": "pending", "reason": "execution_fields_unavailable"}

    bnr_config = load_bnr_config()
    controller = dict(controller_state or {})
    accepted = dict(search_results.get("accepted_trial", {}) or {})
    overrides = dict(accepted.get("overrides", {}) or {})
    threshold = float(
        overrides.get(
            "decision_threshold",
            controller.get(
                "frozen_threshold",
                bnr_config.get("frozen_benchmark", {}).get("threshold", 0.45),
            ),
        )
        or 0.45
    )
    sizing_policy = str(
        overrides.get(
            "sizing_policy",
            controller.get(
                "benchmark_sizing_policy",
                bnr_config.get("frozen_benchmark", {}).get(
                    "sizing_policy", "binary_threshold_v1"
                ),
            ),
        )
    )
    regime_throttle_policy = str(
        overrides.get(
            "regime_throttle_policy",
            controller.get(
                "benchmark_regime_throttle_policy",
                bnr_config.get("frozen_benchmark", {}).get(
                    "regime_throttle_policy", "none"
                ),
            ),
        )
    )
    regime_size_policy = str(
        overrides.get(
            "regime_size_policy",
            controller.get(
                "benchmark_regime_size_policy",
                bnr_config.get("frozen_benchmark", {}).get(
                    "regime_size_policy", "none"
                ),
            ),
        )
    )

    actual = run_event_driven_policy_backtest(
        frame.to_dict(orient="records"),
        threshold=threshold,
        sizing_policy=sizing_policy,
        regime_throttle_policy=regime_throttle_policy,
        regime_size_policy=regime_size_policy,
    )
    if actual.get("status") != "complete":
        return {
            "status": "pending",
            "reason": "actual_execution_unavailable",
            "execution_status": actual.get("status"),
        }

    rng = random.Random(42)
    shuffled = frame.copy()
    probabilities = shuffled["probability"].tolist()
    rng.shuffle(probabilities)
    shuffled["probability"] = probabilities
    random_exec = run_event_driven_policy_backtest(
        shuffled.to_dict(orient="records"),
        threshold=threshold,
        sizing_policy=sizing_policy,
        regime_throttle_policy=regime_throttle_policy,
        regime_size_policy=regime_size_policy,
    )
    if random_exec.get("status") != "complete":
        return {
            "status": "pass",
            "reason": "random_baseline_did_not_clear_threshold",
            "actual_trade_count": int(actual.get("trade_count", 0) or 0),
            "random_execution_status": random_exec.get("status"),
        }

    actual_utility = compute_execution_utility(actual)
    random_utility = compute_execution_utility(random_exec)
    actual_total_pnl = float(actual.get("total_pnl_r", 0.0) or 0.0)
    random_total_pnl = float(random_exec.get("total_pnl_r", 0.0) or 0.0)
    actual_utility_score = actual_utility.get("score")
    random_utility_score = random_utility.get("score")
    utility_delta = (
        float(actual_utility_score) - float(random_utility_score)
        if actual_utility_score is not None and random_utility_score is not None
        else None
    )
    status = (
        "pass"
        if actual_total_pnl > random_total_pnl
        and (utility_delta is None or utility_delta > 0)
        else "fail"
    )
    return {
        "status": status,
        "method": "probability_shuffle_control",
        "threshold": threshold,
        "actual_trade_count": int(actual.get("trade_count", 0) or 0),
        "random_trade_count": int(random_exec.get("trade_count", 0) or 0),
        "actual_total_pnl_r": actual_total_pnl,
        "random_total_pnl_r": random_total_pnl,
        "actual_utility_score": actual_utility_score,
        "random_utility_score": random_utility_score,
        "utility_delta": utility_delta,
    }


def _overfitting_check(
    walk_forward: dict[str, Any],
    cpcv: dict[str, Any],
    multiple_testing: dict[str, Any],
    deflated_sharpe: dict[str, Any],
) -> str:
    if (
        walk_forward.get("status") == "fail"
        or cpcv.get("status") == "fail"
        or multiple_testing.get("status") == "fail"
        or deflated_sharpe.get("status") == "fail"
    ):
        return "fail"
    if (
        walk_forward.get("status") == "pass"
        and cpcv.get("status") == "pass"
        and multiple_testing.get("status") == "pass"
        and deflated_sharpe.get("status") == "pass"
    ):
        return "pass"
    return "pending"


def _centered_trial_deltas(trial_deltas: list[float]) -> list[float]:
    if not trial_deltas:
        return []
    mean_delta = sum(trial_deltas) / len(trial_deltas)
    return [delta - mean_delta for delta in trial_deltas]


def _familywise_max_pvalue(
    centered_deltas: list[float], *, observed_best: float, bootstrap_samples: int = 2000
) -> float:
    if observed_best <= 0:
        return 1.0
    if not centered_deltas:
        return 1.0
    if len(centered_deltas) == 1:
        return 0.0
    rng = random.Random(42)
    exceedances = 0
    sample_size = len(centered_deltas)
    for _ in range(bootstrap_samples):
        sample_max = max(rng.choice(centered_deltas) for _ in range(sample_size))
        if sample_max >= observed_best:
            exceedances += 1
    return (exceedances + 1) / (bootstrap_samples + 1)


def _empirical_effect_floor(trial_deltas: list[float]) -> float:
    positive = sorted(delta for delta in trial_deltas if delta > 0)
    if not positive:
        return 0.0
    if len(positive) == 1:
        return positive[0] * 0.5
    idx = max(0, int(0.75 * (len(positive) - 1)))
    return positive[idx]
