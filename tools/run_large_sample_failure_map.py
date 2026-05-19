from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.break_quality_policy import apply_break_quality_policy
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.failure_analysis import build_failure_map
from trading_ml.paths import DATA_DIR
from trading_ml.stage2_data import load_ohlcv_file
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.validation_audit import build_validation_audit


def _build_session_slice(source_path: str, *, max_sessions: int, symbol: str, timeframe: str, timezone: str) -> str:
    bars = load_ohlcv_file(source_path, symbol=symbol, timeframe=timeframe, timezone=timezone)
    session_dates = sorted({idx.date() for idx in bars.index})
    chosen = set(session_dates[:max_sessions])
    subset = bars[[session_date in chosen for session_date in bars.index.date]].copy()
    output = DATA_DIR / "cache" / f"{symbol.lower()}_{timeframe}_failure_map_slice_{max_sessions}sessions.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output)
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a larger-sample failure map for the current baseline research stack.")
    parser.add_argument("--max-sessions", type=int, default=160)
    parser.add_argument("--feature-family", default="")
    parser.add_argument("--gate-policy", default="")
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
    if args.feature_family:
        stage2_config["feature_family"] = args.feature_family

    result = run_stage2_research_engine(Stage2Config(**stage2_config))
    validation = build_validation_audit(result, {})
    stitched = list(validation.get("walk_forward", {}).get("stitched_prediction_records", []))
    if args.gate_policy:
        selected_records = apply_break_quality_policy(
            stitched,
            result.get("features_records", []),
            policy_name=args.gate_policy,
            threshold=args.threshold,
        )
        execution = run_event_driven_policy_backtest(selected_records, threshold=0.0)
    else:
        selected_records = stitched
        execution = run_event_driven_policy_backtest(selected_records, threshold=args.threshold)
    failure_map = build_failure_map(result, selected_records, execution)

    payload = {
        "source": "large_sample_failure_map",
        "max_sessions": args.max_sessions,
        "source_path": stage2_config["source_path"],
        "baseline": {
            "feature_family": stage2_config.get("feature_family"),
            "threshold": args.threshold,
            "gate_policy": args.gate_policy or "none",
            "setup_subtype": stage2_config.get("setup_subtype", "all_subtypes"),
        },
        "walk_forward": validation.get("walk_forward", {}),
        "execution": {
            "trade_count": execution.get("trade_count"),
            "total_pnl_r": execution.get("total_pnl_r"),
            "avg_trade_r": execution.get("avg_trade_r"),
            "max_drawdown_r": execution.get("max_drawdown_r"),
        },
        "failure_map": failure_map,
    }
    output = Path("reports/large_sample_failure_map.json")
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
