from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.config import load_bnr_config
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.translation_analysis import build_translation_analysis
from trading_ml.utility_analysis import compute_execution_utility
from trading_ml.validation_audit import build_validation_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-protocol confirmation for the frozen or overridden benchmark.")
    parser.add_argument("--feature-family", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--benchmark-name", default=None)
    args = parser.parse_args()

    state = build_agent_loop_state()
    bnr_config = load_bnr_config()
    benchmark = dict(bnr_config.get("frozen_benchmark", {}))
    if args.feature_family:
        benchmark["feature_family"] = args.feature_family
        state["stage2_config"]["feature_family"] = args.feature_family
    if args.threshold is not None:
        benchmark["threshold"] = float(args.threshold)
    benchmark_name = args.benchmark_name or str(bnr_config.get("controller", {}).get("benchmark_name", "bnr_benchmark"))

    result = run_stage2_research_engine(Stage2Config(**state["stage2_config"]))
    validation = build_validation_audit(result, {})
    stitched_records = list(validation.get("walk_forward", {}).get("stitched_prediction_records", []))
    translation = build_translation_analysis(result, prediction_records=stitched_records)
    execution_backtest = run_event_driven_policy_backtest(stitched_records, threshold=float(benchmark.get("threshold", 0.5)))
    utility = compute_execution_utility(execution_backtest) if execution_backtest.get("status") == "complete" else {"score": None}

    applied_threshold = float(benchmark.get("threshold", 0.5))
    threshold_row = next(
        (row for row in translation.get("rows", []) if float(row.get("threshold", -1.0)) == applied_threshold),
        None,
    )

    summary = {
        "benchmark_name": benchmark_name,
        "benchmark": benchmark,
        "stage2_config": result.get("config", {}),
        "candidate_count": result.get("candidate_count", 0),
        "label_summary": result.get("label_summary", {}),
        "model_summary": {
            "status": result.get("model_summary", {}).get("status"),
            "metrics": result.get("model_summary", {}).get("metrics", {}),
        },
        "validation": validation,
        "translation": {
            "source": "walk_forward_stitched" if stitched_records else "holdout_split",
            "stitched_trade_candidates": len(stitched_records),
            "applied_threshold": applied_threshold,
            "applied_threshold_row": threshold_row,
            "best_threshold": translation.get("best_threshold"),
            "status": translation.get("status"),
        },
        "execution_backtest": execution_backtest,
        "utility": utility,
        "confirmation_status": _confirmation_status(validation, threshold_row, execution_backtest),
    }

    output_path = Path("reports") / f"{benchmark_name}_confirmation.json"
    payload = json.dumps(summary, indent=2, sort_keys=True, default=str)
    output_path.write_text(payload, encoding="utf-8")
    print(payload)


def _confirmation_status(validation: dict, threshold_row: dict | None, execution_backtest: dict) -> str:
    walk_forward = dict(validation.get("walk_forward", {}))
    purging = dict(validation.get("purging", {}))
    if walk_forward.get("status") != "pass":
        return "freeze"
    if purging.get("status") != "pass":
        return "freeze"
    if not threshold_row:
        return "freeze"
    if int(threshold_row.get("trade_count", 0) or 0) <= 0:
        return "freeze"
    if float(threshold_row.get("avg_pnl_r", 0.0) or 0.0) <= 0:
        return "freeze"
    if execution_backtest.get("status") != "complete":
        return "freeze"
    if float(execution_backtest.get("total_pnl_r", 0.0) or 0.0) <= 0:
        return "freeze"
    return "confirmed"


if __name__ == "__main__":
    main()
