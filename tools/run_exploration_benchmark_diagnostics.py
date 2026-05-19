from __future__ import annotations

import json
from pathlib import Path

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.diagnostic_adapter import prepare_diagnostic_runtime
from trading_ml.paths import REPORTS_DIR
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.validation_audit import build_validation_audit


def main() -> None:
    prepare_diagnostic_runtime()
    state = build_agent_loop_state()
    config = Stage2Config(**state["stage2_config"])
    result = run_stage2_research_engine(config)
    audit = build_validation_audit(result, {})
    shap_analysis = result.get("model_diagnostics", {}).get("shap_analysis", {})
    payload = {
        "source": "exploration_benchmark_diagnostics",
        "source_path": config.source_path,
        "feature_family": config.feature_family,
        "model_family": config.model_family,
        "threshold": state.get("bnr_spec", {}).get("frozen_benchmark", {}).get("threshold", 0.45),
        "walk_forward": {
            "status": audit["walk_forward"].get("status"),
            "mean_roc_auc": audit["walk_forward"].get("mean_roc_auc"),
            "fold_count": audit["walk_forward"].get("fold_count"),
        },
        "cpcv": {
            "status": audit["cpcv"].get("status"),
            "pbo": audit["cpcv"].get("pbo"),
            "mean_total_pnl_r": audit["cpcv"].get("mean_total_pnl_r"),
            "mean_roc_auc": audit["cpcv"].get("mean_roc_auc"),
            "evaluated_paths": audit["cpcv"].get("evaluated_paths"),
        },
        "overfitting": audit.get("overfitting"),
        "shap_analysis": {
            "status": shap_analysis.get("status"),
            "top_features": shap_analysis.get("top_features", [])[:10],
            "worst_trade_explanations": shap_analysis.get("worst_trade_explanations", [])[:5],
        },
    }
    output = REPORTS_DIR / "exploration_benchmark_diagnostics.json"
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
