from __future__ import annotations

import json
from pathlib import Path

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.config import load_bnr_config
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.utility_analysis import compute_execution_utility
from trading_ml.validation_audit import build_validation_audit


def main() -> None:
    state = build_agent_loop_state()
    bnr_config = load_bnr_config()
    benchmark = dict(bnr_config.get("frozen_benchmark", {}))
    benchmark_name = str(
        bnr_config.get("controller", {}).get("benchmark_name", "bnr_benchmark")
    )
    threshold = float(benchmark.get("threshold", 0.6))
    feature_families = [
        benchmark.get("feature_family", "pre_trigger_context"),
        *list(
            bnr_config.get("feature_search_v1", {})
            .get("space", {})
            .get("feature_family", [])
        ),
    ]

    rows: list[dict] = []
    seen: set[str] = set()
    for family in feature_families:
        if family in seen:
            continue
        seen.add(str(family))
        config = dict(state["stage2_config"])
        config["feature_family"] = str(family)
        result = run_stage2_research_engine(Stage2Config(**config))
        validation = build_validation_audit(result, {})
        stitched_records = list(
            validation.get("walk_forward", {}).get("stitched_prediction_records", [])
        )
        execution = run_event_driven_policy_backtest(
            stitched_records, threshold=threshold
        )
        utility = (
            compute_execution_utility(execution)
            if execution.get("status") == "complete"
            else {"score": None}
        )
        rows.append(
            {
                "feature_family": str(family),
                "trade_count": int(execution.get("trade_count", 0) or 0),
                "total_pnl_r": float(execution.get("total_pnl_r", 0.0) or 0.0),
                "avg_trade_r": float(execution.get("avg_trade_r", 0.0) or 0.0),
                "max_drawdown_r": float(execution.get("max_drawdown_r", 0.0) or 0.0),
                "win_rate": float(execution.get("win_rate", 0.0) or 0.0),
                "positive_sessions": int(execution.get("positive_sessions", 0) or 0),
                "negative_sessions": int(execution.get("negative_sessions", 0) or 0),
                "utility_score": utility.get("score"),
                "status": execution.get("status"),
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["utility_score"] or float("-inf")),
            row["total_pnl_r"],
        ),
        reverse=True,
    )
    payload = {
        "benchmark_name": benchmark_name,
        "threshold": threshold,
        "source": "walk_forward_stitched_event_driven_feature_cycle",
        "ranked_rows": ranked,
        "best_feature_family": ranked[0] if ranked else None,
    }
    output_path = Path("reports") / f"{benchmark_name}_event_feature_cycle.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
