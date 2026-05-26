from __future__ import annotations

import json
import os

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.diagnostic_adapter import prepare_diagnostic_runtime
from trading_ml.paths import REPORTS_DIR
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.translation_analysis import build_translation_analysis
from trading_ml.validation_audit import build_validation_audit


def main() -> None:
    prepare_diagnostic_runtime()
    os.environ.setdefault("TRADING_ML_DISABLE_SHAP", "1")
    state = build_agent_loop_state()
    config = Stage2Config(**state["stage2_config"])
    result = run_stage2_research_engine(config)
    audit = build_validation_audit(
        result,
        {},
        state.get("controller_state", {}),
        artifact_context={"run_id": state.get("run_id")},
    )
    shap_analysis = result.get("model_diagnostics", {}).get("shap_analysis", {})
    stitched = list(
        audit.get("walk_forward", {}).get("stitched_prediction_records", [])
    )
    translation = build_translation_analysis(
        result,
        prediction_records=stitched or None,
        sizing_policy=state.get("controller_state", {}).get("benchmark_sizing_policy"),
        regime_throttle_policy=state.get("controller_state", {}).get(
            "benchmark_regime_throttle_policy"
        ),
        regime_size_policy=state.get("controller_state", {}).get(
            "benchmark_regime_size_policy"
        ),
    )
    best_translation = dict(translation.get("best_threshold", {}))
    payload = {
        "source": "exploration_benchmark_diagnostics",
        "source_path": config.source_path,
        "feature_family": config.feature_family,
        "model_family": config.model_family,
        "threshold": state.get("bnr_spec", {})
        .get("frozen_benchmark", {})
        .get("threshold", 0.45),
        "walk_forward": {
            "status": audit["walk_forward"].get("status"),
            "mean_roc_auc": audit["walk_forward"].get("mean_roc_auc"),
            "fold_count": audit["walk_forward"].get("fold_count"),
        },
        "cpcv": {
            "status": audit["cpcv"].get("status"),
            "artifact_root": audit["cpcv"].get("artifact_root"),
            "pbo": audit["cpcv"].get("pbo"),
            "mean_total_pnl_r": audit["cpcv"].get("mean_total_pnl_r"),
            "median_total_pnl_r": audit["cpcv"].get("median_total_pnl_r"),
            "min_path_pnl_r": audit["cpcv"].get("min_path_pnl_r"),
            "path_positive_rate": audit["cpcv"].get("path_positive_rate"),
            "mean_roc_auc": audit["cpcv"].get("mean_roc_auc"),
            "evaluated_paths": audit["cpcv"].get("evaluated_paths"),
            "worst_paths": audit["cpcv"].get("worst_paths", []),
            "best_paths": audit["cpcv"].get("best_paths", []),
            "distribution": audit["cpcv"].get("distribution", {}),
        },
        "deflated_sharpe": audit.get("deflated_sharpe", {}),
        "translation": {
            "status": translation.get("status"),
            "sizing_policy": translation.get("sizing_policy"),
            "regime_throttle_policy": translation.get("regime_throttle_policy"),
            "regime_size_policy": translation.get("regime_size_policy"),
            "best_row": best_translation,
            "utility_gap_vs_binary": best_translation.get("utility_gap_vs_binary"),
            "binary_utility_score": best_translation.get("binary_utility_score"),
            "sized_utility_score": best_translation.get("utility_score"),
        },
        "overfitting": audit.get("overfitting"),
        "shap_analysis": {
            "status": shap_analysis.get("status"),
            "top_features": shap_analysis.get("top_features", [])[:10],
            "worst_trade_explanations": shap_analysis.get(
                "worst_trade_explanations", []
            )[:5],
        },
    }
    output = REPORTS_DIR / "exploration_benchmark_diagnostics.json"
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
