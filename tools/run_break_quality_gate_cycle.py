from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.break_quality_policy import apply_break_quality_policy, get_break_quality_policies
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.paths import DATA_DIR
from trading_ml.stage2_data import load_ohlcv_file
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.utility_analysis import compute_execution_utility
from trading_ml.validation_audit import build_validation_audit


def _build_session_slice(source_path: str, *, max_sessions: int, symbol: str, timeframe: str, timezone: str) -> str:
    bars = load_ohlcv_file(source_path, symbol=symbol, timeframe=timeframe, timezone=timezone)
    session_dates = sorted({idx.date() for idx in bars.index})
    chosen = set(session_dates[:max_sessions])
    subset = bars[[session_date in chosen for session_date in bars.index.date]].copy()
    output = DATA_DIR / "cache" / f"{symbol.lower()}_{timeframe}_break_gate_slice_{max_sessions}sessions.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output)
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a break-quality gate cycle on the 160-session floor.")
    parser.add_argument("--max-sessions", type=int, default=160)
    parser.add_argument("--feature-family", default="bnr_plus_context")
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()

    state = build_agent_loop_state()
    stage2_config = dict(state["stage2_config"])
    if args.max_sessions > 0:
        stage2_config["source_path"] = _build_session_slice(
            stage2_config["source_path"],
            max_sessions=args.max_sessions,
            symbol=stage2_config["symbol"],
            timeframe=stage2_config["timeframe"],
            timezone=stage2_config["timezone"],
        )
    stage2_config["feature_family"] = args.feature_family

    result = run_stage2_research_engine(Stage2Config(**stage2_config))
    validation = build_validation_audit(result, {})
    stitched = list(validation.get("walk_forward", {}).get("stitched_prediction_records", []))

    rows: list[dict] = []
    output = Path("reports/break_quality_gate_cycle.json")
    for policy in get_break_quality_policies():
        filtered = apply_break_quality_policy(
            stitched,
            result.get("features_records", []),
            policy_name=policy["name"],
            threshold=args.threshold,
        )
        execution = run_event_driven_policy_backtest(filtered, threshold=0.0)
        utility = compute_execution_utility(execution) if execution.get("status") == "complete" else {"score": None}
        rows.append(
            {
                "policy_name": policy["name"],
                "selected_signal_count": len(filtered),
                "trade_count": int(execution.get("trade_count", 0) or 0),
                "total_pnl_r": float(execution.get("total_pnl_r", 0.0) or 0.0),
                "avg_trade_r": float(execution.get("avg_trade_r", 0.0) or 0.0),
                "win_rate": float(execution.get("win_rate", 0.0) or 0.0),
                "max_drawdown_r": float(execution.get("max_drawdown_r", 0.0) or 0.0),
                "utility_score": utility.get("score"),
                "walk_forward_status": validation.get("walk_forward", {}).get("status"),
                "walk_forward_mean_roc_auc": validation.get("walk_forward", {}).get("mean_roc_auc"),
            }
        )
        output.write_text(json.dumps({"partial_rows": rows}, indent=2, default=str), encoding="utf-8")

    ranked = sorted(rows, key=lambda row: (float(row["utility_score"] or float("-inf")), row["total_pnl_r"]), reverse=True)
    payload = {
        "source": "governed_break_quality_gate_cycle",
        "max_sessions": args.max_sessions,
        "feature_family": args.feature_family,
        "threshold": args.threshold,
        "source_path": stage2_config["source_path"],
        "ranked_rows": ranked,
        "best_row": ranked[0] if ranked else None,
    }
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
