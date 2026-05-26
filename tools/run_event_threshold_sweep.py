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
    thresholds = list(
        bnr_config.get("translation_contract", {}).get(
            "threshold_grid", [0.45, 0.5, 0.55, 0.6, 0.65]
        )
    )
    benchmark_name = str(
        bnr_config.get("controller", {}).get("benchmark_name", "bnr_benchmark")
    )

    result = run_stage2_research_engine(Stage2Config(**state["stage2_config"]))
    validation = build_validation_audit(result, {})
    stitched_records = list(
        validation.get("walk_forward", {}).get("stitched_prediction_records", [])
    )

    rows: list[dict] = []
    for threshold in thresholds:
        summary = run_event_driven_policy_backtest(
            stitched_records, threshold=float(threshold)
        )
        utility = (
            compute_execution_utility(summary)
            if summary.get("status") == "complete"
            else {"score": None}
        )
        rows.append(
            {
                "threshold": float(threshold),
                "status": summary.get("status"),
                "signal_count": int(summary.get("signal_count", 0) or 0),
                "trade_count": int(summary.get("trade_count", 0) or 0),
                "overlap_skips": int(summary.get("overlap_skips", 0) or 0),
                "total_pnl_r": float(summary.get("total_pnl_r", 0.0) or 0.0),
                "avg_trade_r": float(summary.get("avg_trade_r", 0.0) or 0.0),
                "win_rate": float(summary.get("win_rate", 0.0) or 0.0),
                "max_drawdown_r": float(summary.get("max_drawdown_r", 0.0) or 0.0),
                "session_count": int(summary.get("session_count", 0) or 0),
                "positive_sessions": int(summary.get("positive_sessions", 0) or 0),
                "negative_sessions": int(summary.get("negative_sessions", 0) or 0),
                "utility_score": utility.get("score"),
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["utility_score"] or float("-inf")),
            row["total_pnl_r"],
            row["avg_trade_r"],
        ),
        reverse=True,
    )
    payload = {
        "benchmark_name": benchmark_name,
        "source": "walk_forward_stitched_event_driven",
        "threshold_grid": thresholds,
        "ranked_rows": ranked,
        "best_threshold": ranked[0] if ranked else None,
    }

    output_path = Path("reports") / f"{benchmark_name}_event_threshold_sweep.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
