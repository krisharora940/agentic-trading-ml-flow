from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_ml.agent_workflow import build_agent_loop_state
from trading_ml.config import load_bnr_config
from trading_ml.paths import DATA_DIR
from trading_ml.event_driven_backtest import run_event_driven_policy_backtest
from trading_ml.stage2_data import load_ohlcv_file
from trading_ml.stage2_pipeline import Stage2Config, run_stage2_research_engine
from trading_ml.utility_analysis import compute_execution_utility
from trading_ml.validation_audit import build_validation_audit


def _build_session_slice(
    source_path: str, *, max_sessions: int, symbol: str, timeframe: str, timezone: str
) -> str:
    bars = load_ohlcv_file(
        source_path, symbol=symbol, timeframe=timeframe, timezone=timezone
    )
    session_dates = sorted({idx.date() for idx in bars.index})
    chosen = set(session_dates[:max_sessions])
    mask = [session_date in chosen for session_date in bars.index.date]
    subset = bars[mask].copy()
    output = (
        DATA_DIR
        / "cache"
        / f"{symbol.lower()}_{timeframe}_subtype_slice_{max_sessions}sessions.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output)
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a governed BNR subtype cycle on the active Stage 2 source or a sliced subset."
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="If set, build a smaller parquet slice using the first N sessions.",
    )
    args = parser.parse_args()

    state = build_agent_loop_state()
    config = load_bnr_config()
    subtypes = list(config["subtype_search_v1"]["space"]["setup_subtype"])
    threshold = float(config["frozen_benchmark"]["threshold"])
    output = Path("reports/subtype_cycle.json")
    stage2_config = dict(state["stage2_config"])
    if args.max_sessions > 0:
        stage2_config["source_path"] = _build_session_slice(
            stage2_config["source_path"],
            max_sessions=args.max_sessions,
            symbol=stage2_config["symbol"],
            timeframe=stage2_config["timeframe"],
            timezone=stage2_config["timezone"],
        )

    rows: list[dict] = []
    for subtype in subtypes:
        trial = dict(stage2_config)
        trial["setup_subtype"] = subtype
        result = run_stage2_research_engine(Stage2Config(**trial))
        validation = build_validation_audit(result, {})
        stitched = list(
            validation.get("walk_forward", {}).get("stitched_prediction_records", [])
        )
        execution = run_event_driven_policy_backtest(stitched, threshold=threshold)
        utility = (
            compute_execution_utility(execution)
            if execution.get("status") == "complete"
            else {"score": None}
        )
        rows.append(
            {
                "setup_subtype": subtype,
                "candidate_count": int(result.get("candidate_count", 0)),
                "subtype_counts": result.get("subtype_counts", {}),
                "walk_forward_status": validation.get("walk_forward", {}).get("status"),
                "walk_forward_mean_roc_auc": validation.get("walk_forward", {}).get(
                    "mean_roc_auc"
                ),
                "trade_count": int(execution.get("trade_count", 0) or 0),
                "total_pnl_r": float(execution.get("total_pnl_r", 0.0) or 0.0),
                "avg_trade_r": float(execution.get("avg_trade_r", 0.0) or 0.0),
                "max_drawdown_r": float(execution.get("max_drawdown_r", 0.0) or 0.0),
                "utility_score": utility.get("score"),
                "market_structure_lab": result.get("market_structure_lab", {}),
            }
        )
        output.write_text(
            json.dumps({"partial_rows": rows}, indent=2, default=str), encoding="utf-8"
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
        "source": "governed_subtype_cycle",
        "max_sessions": args.max_sessions,
        "source_path": stage2_config["source_path"],
        "ranked_rows": ranked,
        "best_row": ranked[0] if ranked else None,
    }
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
